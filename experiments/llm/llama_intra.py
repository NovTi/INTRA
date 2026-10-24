"""
llama_intra.py -- INTRA static sparse attention for LLaMA-3.1 / Qwen3 as a transformers (5.x)
attention backend.

Each decoder layer is assigned one of three branches by a cycled `layer_pattern`:

  * S (Scatter)  block-level Scatter. Position i belongs to group (i // block) % num_groups and
                 attends causally to the members of its group. With `scatter_stripe`, the s-th
                 Scatter layer is shifted by an offset so that a query block in group g reads key
                 blocks of group (g + offset) % num_groups, plus its own block.
  * W (Window)   causal sliding window of `window` keys (FlexAttention).
  * D (Dense)    full causal attention.

Both S and W layers optionally keep the first `n_sink` tokens visible to every query
(StreamingLLM-style attention sinks).

The paper's LLaMA configuration ("g2s4") is layer_pattern="SWWWWWWW", num_groups=2,
scatter_blocks=(256,), scatter_stripe=True, scatter_n_offsets=2, window=575, n_sink=64: four
Scatter layers (0, 8, 16, 24) that each read every second 256-token block, and 28 window layers.

Usage: import this module (it registers attn_implementation="intra"), load the model with
attn_implementation="intra", then call `install_intra_llama(model, ...)`, which stamps the
per-layer config onto each attention module. Sparse attention is applied during prefill only;
single-token decode steps attend densely to the KV cache.

SDPA + FlexAttention only, both with a real backward pass.
"""
import torch
import torch.nn.functional as F
from torch.nn.attention.flex_attention import flex_attention, create_block_mask

try:
    from transformers import AttentionInterface
except Exception:  # older name
    AttentionInterface = None
from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS


# ---- FlexAttention block-mask caches, keyed by shape/config/device ----
_BLOCKMASK_CACHE = {}
_FLEX = None


def _get_flex(compile_flex: bool):
    """The compiled/uncompiled choice is cached globally on first use and `compile_flex` is
    ignored afterwards; call reset_flex() to switch modes inside one process. Uncompiled
    FlexAttention materialises the full S x S score matrix and cannot skip masked tiles."""
    global _FLEX
    if _FLEX is None:
        _FLEX = torch.compile(flex_attention) if compile_flex else flex_attention
    return _FLEX


def reset_flex():
    """Drop the cached flex callable so the next _get_flex() honours its argument again."""
    global _FLEX
    _FLEX = None


def _sliding_block_mask(S, window, device, n_sink=0):
    """Causal sliding window, optionally with attention sinks: the first `n_sink` tokens stay
    visible to every query."""
    key = (S, window, n_sink, str(device))
    bm = _BLOCKMASK_CACHE.get(key)
    if bm is None:
        if n_sink > 0:
            def mask_mod(b, h, q_idx, kv_idx):
                causal = kv_idx <= q_idx
                local = (q_idx - kv_idx) < window
                sink = kv_idx < n_sink
                return causal & (local | sink)
        else:
            def mask_mod(b, h, q_idx, kv_idx):
                return (kv_idx <= q_idx) & (q_idx - kv_idx < window)
        # B=None,H=None -> shared across batch/heads
        bm = create_block_mask(mask_mod, B=None, H=None, Q_LEN=S, KV_LEN=S,
                               device=device, _compile=True)
        _BLOCKMASK_CACHE[key] = bm
    return bm


# Scatter is block-sparse at FlexAttention's 128x128 tile granularity only when `block` is at least
# the tile size: then (idx // block) is constant across a tile, so each tile is either fully visible
# or fully masked and 1/num_groups of the tiles are skipped. With block < 128 every tile contains
# some visible pair and FlexAttention runs at dense cost.
_FLEX_MIN_BLOCK = 128
_SCATTER_BM_CACHE = {}


def _scatter_block_mask(S, num_groups, block, offset, n_sink, device):
    key = (S, num_groups, block, offset, n_sink, str(device))
    bm = _SCATTER_BM_CACHE.get(key)
    if bm is None:
        G, blk, off, ns = int(num_groups), int(block), int(offset), int(n_sink)

        def mask_mod(b, h, q_idx, kv_idx):
            causal = kv_idx <= q_idx
            qb, kb = q_idx // blk, kv_idx // blk
            vis = (kb % G) == ((qb + off) % G)
            if off:
                # The own block is mandatory: with off != 0 a query in the first block has no
                # causal key in its target residue class, so its row would be fully masked (NaN).
                vis = vis | (kb == qb)
            if ns:
                vis = vis | (kv_idx < ns)
            return causal & vis

        bm = create_block_mask(mask_mod, B=None, H=None, Q_LEN=S, KV_LEN=S,
                               device=device, _compile=True)
        _SCATTER_BM_CACHE[key] = bm
    return bm


def _scatter_flex(q, k, v, num_groups, scaling, n_sink, block, offset, compile_flex):
    """Block-aligned Scatter via FlexAttention. Same math as _scatter_attention at offset=0, but
    block-sparse at the kernel level instead of num_groups gathered SDPA calls."""
    bm = _scatter_block_mask(q.shape[2], num_groups, block, offset, n_sink, q.device)
    flex = _get_flex(compile_flex)
    return flex(q, k, v, block_mask=bm, scale=scaling, enable_gqa=(k.shape[1] != q.shape[1]))


def _scatter_attention(q, k, v, num_groups, scaling, n_sink=0, block=1):
    """q:(B,Hq,S,D)  k,v:(B,Hkv,S,D). Causal within each Scatter group.

    Group assignment is `(pos // block) % num_groups`, so positions i and j are directly
    connected iff (i//block) == (j//block) (mod num_groups). block=1 is the element-strided
    `i % num_groups`.

    With n_sink>0, the first `n_sink` positions are appended to every group's KV set so all
    queries retain access to the attention-sink tokens (they otherwise only live in group 0).
    Sinks are excluded from the group's own KV list to avoid double-counting; sink positions are
    still queries of their own group.
    """
    B, Hq, S, D = q.shape
    out = torch.zeros_like(q)
    idx_all = torch.arange(S, device=q.device)
    gqa = (k.shape[1] != Hq)
    gid = (idx_all // max(1, int(block))) % num_groups
    for i in range(num_groups):
        idx = idx_all[gid == i]
        if idx.numel() == 0:
            continue
        if n_sink <= 0:
            oi = F.scaled_dot_product_attention(
                q[:, :, idx, :], k[:, :, idx, :], v[:, :, idx, :],
                is_causal=True, scale=scaling, enable_gqa=gqa)
        else:
            sink = idx_all[:n_sink]
            gidx = idx[idx >= n_sink]                      # group members that are not sinks
            kv_idx = torch.cat([sink, gidx])
            # explicit mask: sinks visible where causal allows, group part causal in original
            # positions. Every row has at least kv=sink[0]<=q, so no row is fully masked.
            m = kv_idx.view(1, -1) <= idx.view(-1, 1)
            oi = F.scaled_dot_product_attention(
                q[:, :, idx, :], k[:, :, kv_idx, :], v[:, :, kv_idx, :],
                attn_mask=m.view(1, 1, idx.numel(), kv_idx.numel()),
                scale=scaling, enable_gqa=gqa)
        out[:, :, idx, :] = oi
    return out  # (B,Hq,S,D)


def _window_attention(q, k, v, window, scaling, compile_flex, n_sink=0):
    """Causal sliding-window via FlexAttention. q:(B,Hq,S,D) k,v:(B,Hkv,S,D)."""
    B, Hq, S, D = q.shape
    bm = _sliding_block_mask(S, window, q.device, n_sink)
    flex = _get_flex(compile_flex)
    out = flex(q, k, v, block_mask=bm, scale=scaling, enable_gqa=(k.shape[1] != Hq))
    return out  # (B,Hq,S,D)


def _dense_causal(q, k, v, scaling):
    Hq = q.shape[1]
    return F.scaled_dot_product_attention(q, k, v, is_causal=True, scale=scaling,
                                          enable_gqa=(k.shape[1] != Hq))


def intra_attention_forward(module, query, key, value, attention_mask=None,
                            scaling=None, dropout=0.0, **kwargs):
    """transformers attention-interface fn. q/k/v: (B, H, S, D). Returns (attn_out (B,S,Hq,D), None).

    Per-layer routing via the dict stamped on module._intra by install_intra_llama.
    """
    cfg = getattr(module, "_intra", None)
    if scaling is None:
        scaling = getattr(module, "scaling", query.shape[-1] ** -0.5)
    if query.shape[2] == 1 and key.shape[2] > 1:
        # Decode: INTRA is a prefill-only pattern, so a single query attends to the whole KV cache.
        # is_causal must be False here -- with q_len=1 and kv_len=N, SDPA aligns the causal mask
        # to the top-left, which would leave the query attending to key 0 alone.
        Hq, Hkv = query.shape[1], key.shape[1]
        k2, v2 = key, value
        if Hq != Hkv:
            rep = Hq // Hkv
            k2 = key.repeat_interleave(rep, dim=1)
            v2 = value.repeat_interleave(rep, dim=1)
        am = attention_mask[:, :, :, : k2.shape[-2]] if attention_mask is not None else None
        out = F.scaled_dot_product_attention(query, k2, v2, attn_mask=am, scale=scaling,
                                             is_causal=False)
    elif cfg is None or cfg.get("dense_debug", False):
        out = _dense_causal(query, key, value, scaling)
    else:
        kind = cfg["kind"]
        if kind == "S":
            blk, off = cfg.get("scatter_block", 1), cfg.get("scatter_offset", 0)
            # A non-zero stripe offset can only be expressed as a mask, so it forces flex.
            if off or (cfg.get("scatter_impl", "loop") == "flex" and blk >= _FLEX_MIN_BLOCK):
                out = _scatter_flex(query, key, value, cfg["num_groups"], scaling,
                                    cfg.get("n_sink", 0), blk, off, cfg["compile_flex"])
            else:
                out = _scatter_attention(query, key, value, cfg["num_groups"], scaling,
                                         cfg.get("n_sink", 0), blk)
        elif kind == "W":
            out = _window_attention(query, key, value, cfg["window"], scaling,
                                    cfg["compile_flex"], cfg.get("n_sink", 0))
        else:                                             # 'D' -- full causal attention
            out = _dense_causal(query, key, value, scaling)
    # transformers expects (B, S, Hq, D)
    out = out.transpose(1, 2).contiguous()
    return out, None


# register once
if AttentionInterface is not None:
    try:
        AttentionInterface.register("intra", intra_attention_forward)
    except Exception:
        ALL_ATTENTION_FUNCTIONS["intra"] = intra_attention_forward
else:
    ALL_ATTENTION_FUNCTIONS["intra"] = intra_attention_forward


def install_intra_llama(model, num_groups=8, window=512, dense_debug=False, compile_flex=True,
                        n_sink=0, scatter_blocks=(1,), layer_pattern="SSWW",
                        scatter_stripe=False, scatter_impl="loop", scatter_n_offsets=0):
    """Stamp per-layer INTRA config onto each attention module. Model must be loaded with
    attn_implementation='intra'. Returns (n_scatter, n_window, n_dense).

    layer_pattern: letters cycled over the layers -- 'S' Scatter, 'W' window, 'D' full causal.
        Letters other than S/W/D are ignored. Model-agnostic: walks model.model.layers, so
        LLaMA-3.1 (32 layers) and Qwen3 (36 layers) both work; "SWWWWWWW" gives 4 Scatter
        layers on 32 layers and 5 on 36.
    scatter_blocks: block sizes cycled across the Scatter layers. Keep every block
        <= seq_len / num_groups, otherwise fewer than num_groups groups are populated and the
        layer densifies.
    scatter_stripe: give the s-th Scatter layer offset (s % n_offsets), where n_offsets =
        scatter_n_offsets if > 0 else num_groups. A query block in group g then reads key blocks
        of group (g + offset) % num_groups plus its own block. Requires block >= 128.
    scatter_impl: 'loop' (num_groups gathered SDPA calls) or 'flex' (same math via a cached
        block mask, block-sparse cost; only used when block >= 128). Stripe offsets force flex.
    """
    layers = model.model.layers if hasattr(model, "model") else model.layers
    blocks = list(scatter_blocks) or [1]
    pat = [c for c in layer_pattern.upper() if c in "SWD"] or ["S", "S", "W", "W"]
    n = {"S": 0, "W": 0, "D": 0}
    for li, layer in enumerate(layers):
        kind = pat[li % len(pat)]
        blk = blocks[n["S"] % len(blocks)] if kind == "S" else 1
        n_off = scatter_n_offsets if scatter_n_offsets > 0 else num_groups
        off = (n["S"] % n_off) if (kind == "S" and scatter_stripe) else 0
        if off and blk < _FLEX_MIN_BLOCK:
            raise ValueError(f"scatter_stripe needs block >= {_FLEX_MIN_BLOCK} (got {blk}): the "
                             "stripe is expressible only as a block mask, and below the tile size "
                             "a block mask is 100% dense")
        layer.self_attn._intra = dict(
            num_groups=num_groups, window=window, layer_idx=li, kind=kind,
            dense_debug=dense_debug, compile_flex=compile_flex, n_sink=n_sink,
            scatter_block=blk, scatter_offset=off, scatter_impl=scatter_impl)
        n[kind] += 1
    return n["S"], n["W"], n["D"]


def set_dense_debug(model, flag: bool):
    """Teacher/student toggle: flag=True makes every INTRA layer dense causal."""
    layers = model.model.layers if hasattr(model, "model") else model.layers
    for layer in layers:
        if hasattr(layer.self_attn, "_intra"):
            layer.self_attn._intra["dense_debug"] = flag


if __name__ == "__main__":
    # CPU self-test: every branch against an explicit dense-mask oracle in float64.
    torch.manual_seed(0)

    def oracle(q, k, v, keep):
        rep = q.shape[1] // k.shape[1]
        k, v = k.repeat_interleave(rep, 1), v.repeat_interleave(rep, 1)
        return F.scaled_dot_product_attention(q, k, v, attn_mask=keep)

    B, Hq, Hkv, D = 1, 4, 2, 16
    worst = 0.0
    for S, G, blk, ns in [(64, 8, 1, 0), (195, 8, 1, 4), (256, 8, 4, 64), (512, 2, 128, 0),
                          (512, 2, 128, 64), (1000, 3, 128, 64)]:
        q = torch.randn(B, Hq, S, D, dtype=torch.float64)
        k = torch.randn(B, Hkv, S, D, dtype=torch.float64)
        v = torch.randn(B, Hkv, S, D, dtype=torch.float64)
        i = torch.arange(S)
        causal = i[None, :] <= i[:, None]
        sink = (i[None, :] < ns) & causal
        g = (i // blk) % G
        keep = ((g[:, None] == g[None, :]) & causal) | sink
        err = (_scatter_attention(q, k, v, G, None, ns, blk) - oracle(q, k, v, keep)).abs().max()
        worst = max(worst, err.item())
        if blk >= _FLEX_MIN_BLOCK:
            for off in range(G):
                qb, kb = i[:, None] // blk, i[None, :] // blk
                vis = (kb % G) == ((qb + off) % G)
                if off:
                    vis = vis | (kb == qb)
                keep_s = (vis & causal) | sink
                got = _scatter_flex(q, k, v, G, None, ns, blk, off, compile_flex=False)
                worst = max(worst, (got - oracle(q, k, v, keep_s)).abs().max().item())
        for w in (64, 575):
            keep_w = (causal & ((i[:, None] - i[None, :]) < w)) | sink
            got = _window_attention(q, k, v, w, None, compile_flex=False, n_sink=ns)
            worst = max(worst, (got - oracle(q, k, v, keep_w)).abs().max().item())
    print(f"llama_intra self-test: worst abs err vs dense-mask oracle = {worst:.3e}")
    assert worst < 1e-10, worst
    print("PASS")
