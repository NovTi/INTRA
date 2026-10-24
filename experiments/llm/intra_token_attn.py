#!/usr/bin/env python3
"""INTRA's original language-model Scatter/Window attention as a transformers backend, in PyTorch.

This is a reference for the semantics of the token-level CUDA kernel configuration used by the
original SCROLLS fine-tuning setup (see llamafactory_scrolls/), for quality evaluation without the
kernel. Latency claims require the kernel and cannot be made from this file.

  Scatter (masktype 0): group(i) = (i // block) % num_groups, dense causal attention inside each
      group. block=1, num_groups=8 is the original stride-8 interleave (1/8 density).
  Window (masktype 4): a 64-ALIGNED sliding window -- every query in a 64-token chunk shares the
      window start max(chunk_start - 512, 0), so a query sees between 512 and 576 keys. This
      matches the kernel and the pre-baked window mask of the original pipeline.
  Layer schedule: layer_idx % 4 in {0, 1} -> Scatter, {2, 3} -> Window (two Scatter layers, then
      two Window layers, repeating).

Adapters trained with llama_intra.py must be evaluated with llama_intra.py (run_eval.py
--backend intra_llama): its window is a plain per-token window, not this 64-aligned one.
"""
import torch
import torch.nn.functional as F

NUM_GROUPS = 8
WINDOW_BACK = 512
WINDOW_CHUNK = 64


def _expand_kv(q, k, v):
    hq, hkv = q.shape[1], k.shape[1]
    if hq != hkv:
        rep = hq // hkv
        k = k.repeat_interleave(rep, dim=1)
        v = v.repeat_interleave(rep, dim=1)
    return k, v


def scatter_attn(q, k, v, num_groups=NUM_GROUPS, scaling=None, block=1, n_sink=0):
    """group(i) = (i // block) % num_groups, dense causal attention within each group.

    block=1 is the original element-strided interleave and is the default.

    n_sink>0 keeps the first n_sink positions visible to every group (attention sinks), with the
    same semantics as llama_intra._scatter_attention.
    -> [B, S, H, D]
    """
    B, H, S, D = q.shape
    k, v = _expand_kv(q, k, v)
    out = torch.zeros_like(q)
    idx_all = torch.arange(S, device=q.device)
    gid = (idx_all // max(1, int(block))) % num_groups
    for g in range(num_groups):
        idx = idx_all[gid == g]
        if idx.numel() == 0:
            continue
        if n_sink <= 0:
            # positions inside a group keep their relative order, so causality within the
            # subsequence is exactly is_causal=True -- no explicit mask needed
            o = F.scaled_dot_product_attention(
                q[:, :, idx, :], k[:, :, idx, :], v[:, :, idx, :],
                is_causal=True, scale=scaling)
        else:
            sink = idx_all[:n_sink]
            gidx = idx[idx >= n_sink]            # group members that are not already sinks
            kv_idx = torch.cat([sink, gidx])
            m = kv_idx.view(1, -1) <= idx.view(-1, 1)      # causal in ORIGINAL positions
            o = F.scaled_dot_product_attention(
                q[:, :, idx, :], k[:, :, kv_idx, :], v[:, :, kv_idx, :],
                attn_mask=m.view(1, 1, idx.numel(), kv_idx.numel()), scale=scaling)
        out[:, :, idx, :] = o
    return out.transpose(1, 2).contiguous()


def window_attn(q, k, v, back=WINDOW_BACK, chunk=WINDOW_CHUNK, q_block=2048, scaling=None,
                n_sink=0):
    """64-aligned sliding window of at most back+chunk keys. -> [B, S, H, D]

    Evaluated in query blocks so nothing of size [S, S] is ever built: for a block of queries the only
    keys reachable are [max(first_chunk_start - back, 0), q_end), which is q_block + back wide.

    n_sink>0 additionally keeps positions [0, n_sink) visible to every query (the kernel has no
    sinks).
    """
    B, H, S, D = q.shape
    k, v = _expand_kv(q, k, v)
    out = torch.empty_like(q)
    q_block = max(chunk, (q_block // chunk) * chunk)     # keep blocks 64-aligned

    for qs in range(0, S, q_block):
        qe = min(S, qs + q_block)
        qpos = torch.arange(qs, qe, device=q.device)
        lo = max((qs // chunk) * chunk - back, 0)        # earliest key any query in this block sees
        kpos = torch.arange(lo, qe, device=q.device)
        if n_sink > 0 and lo > 0:
            # prepend the sink positions; they are disjoint from [lo, qe) exactly when n_sink <= lo,
            # and clamped otherwise so no key is listed twice
            kpos = torch.cat([torch.arange(min(n_sink, lo), device=q.device), kpos])

        starts = torch.clamp((qpos // chunk) * chunk - back, min=0).unsqueeze(1)   # [C, 1]
        kk = kpos.unsqueeze(0)                                                     # [1, K]
        keep = (kk >= starts) & (kk <= qpos.unsqueeze(1))                          # window AND causal
        if n_sink > 0:
            keep |= (kk < n_sink) & (kk <= qpos.unsqueeze(1))                      # sinks, still causal
        add = torch.zeros(keep.shape, dtype=q.dtype, device=q.device)
        add.masked_fill_(~keep, torch.finfo(q.dtype).min)

        o = F.scaled_dot_product_attention(
            q[:, :, qs:qe, :], k[:, :, kpos, :], v[:, :, kpos, :],
            attn_mask=add.unsqueeze(0).unsqueeze(0), scale=scaling)
        out[:, :, qs:qe, :] = o
    return out.transpose(1, 2).contiguous()


def layer_masktype(layer_idx):
    """Two Scatter layers, then two Window layers, repeating."""
    return 0 if (layer_idx % 4 in (0, 1)) else 4


def make_intra_attention(num_groups=NUM_GROUPS, back=WINDOW_BACK, chunk=WINDOW_CHUNK,
                         force_masktype=None, block=1, n_sink=0):
    """transformers ALL_ATTENTION_FUNCTIONS-compatible callable."""
    def intra_attention_forward(module, query, key, value, attention_mask=None,
                                scaling=None, dropout=0.0, **kwargs):
        if query.shape[2] == 1:          # decode: INTRA is prefill-only, same as MoBA and the kernel
            k2, v2 = _expand_kv(query, key, value)
            am = attention_mask[:, :, :, : k2.shape[-2]] if attention_mask is not None else None
            o = F.scaled_dot_product_attention(query, k2, v2, attn_mask=am,
                                               scale=scaling, is_causal=False)
            return o.transpose(1, 2).contiguous(), None
        mt = force_masktype if force_masktype is not None else layer_masktype(
            getattr(module, "layer_idx", 0))
        if mt == 0:
            return scatter_attn(query, key, value, num_groups, scaling,
                                block=block, n_sink=n_sink), None
        return window_attn(query, key, value, back, chunk, scaling=scaling, n_sink=n_sink), None
    return intra_attention_forward


# ---------------------------------------------------------------- correctness

def _dense_with_mask(q, k, v, keep):
    """Reference: build the full [S, S] boolean mask and run one dense attention."""
    k, v = _expand_kv(q, k, v)
    add = torch.zeros(keep.shape, dtype=q.dtype, device=q.device)
    add.masked_fill_(~keep, torch.finfo(q.dtype).min)
    o = F.scaled_dot_product_attention(q, k, v, attn_mask=add.unsqueeze(0).unsqueeze(0))
    return o.transpose(1, 2).contiguous()


def verify(S=1024, H=4, Hkv=2, D=64, device="cpu", block=1, n_sink=0):
    """Compare both branches against a brute-force [S,S] mask built from the spec, not from this
    file's own logic -- a reference derived from the implementation would agree with any bug."""
    torch.manual_seed(0)
    q = torch.randn(1, H, S, D, device=device, dtype=torch.float32)
    k = torch.randn(1, Hkv, S, D, device=device, dtype=torch.float32)
    v = torch.randn(1, Hkv, S, D, device=device, dtype=torch.float32)
    i = torch.arange(S, device=device)
    causal = i.unsqueeze(1) >= i.unsqueeze(0)

    # Scatter: same group and causal, group = (pos // block) % G
    g = (i // max(1, block)) % NUM_GROUPS
    keep = (g.unsqueeze(1) == g.unsqueeze(0)) & causal
    if n_sink:
        keep = keep | ((i.unsqueeze(0) < n_sink) & causal)
    ref = _dense_with_mask(q, k, v, keep)
    got = scatter_attn(q, k, v, block=block, n_sink=n_sink)
    e1 = (ref - got).abs().max().item() / ref.abs().max().item()

    # Window: the original window-mask generator, transcribed literally
    keep_w = torch.zeros(S, S, dtype=torch.bool, device=device)
    for idx in range(S):
        start = max((idx // WINDOW_CHUNK) * WINDOW_CHUNK - WINDOW_BACK, 0)
        keep_w[idx, start: idx + 1] = True
    if n_sink:
        keep_w = keep_w | ((i.unsqueeze(0) < n_sink) & causal)
    ref_w = _dense_with_mask(q, k, v, keep_w)
    got_w = window_attn(q, k, v, q_block=256, n_sink=n_sink)
    e2 = (ref_w - got_w).abs().max().item() / ref_w.abs().max().item()

    dens_s = keep.float().mean().item() / causal.float().mean().item()
    dens_w = keep_w.float().mean().item() / causal.float().mean().item()
    print(f"  S={S:<5} block={block:<5} sink={n_sink:<3} "
          f"scatter rel_err={e1:.2e} (dens {dens_s:.3f})  window rel_err={e2:.2e} (dens {dens_w:.3f})")
    return max(e1, e2)


if __name__ == "__main__":
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"verifying INTRA Scatter/Window against brute-force dense masks on {dev}")
    cases = [(S, 1, 0) for S in (512, 1024, 2000)] \
        + [(2000, b, 0) for b in (64, 256)] \
        + [(2000, b, s) for b in (1, 256) for s in (4, 64)]
    worst = max(verify(S, device=dev, block=b, n_sink=s) for S, b, s in cases)
    print(f"\nlayer schedule: "
          f"{[layer_masktype(l) for l in range(12)]}")
    print(f"worst relative error {worst:.3e} -> {'PASS' if worst < 1e-4 else 'FAIL'}")
