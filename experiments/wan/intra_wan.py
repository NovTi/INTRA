"""
INTRA 3-D sparse self-attention for Wan2.2-TI2V-5B (reference implementation).

Wan's attn1 is bidirectional self-attention over the video-latent token grid (T, H, W), flattened
T-major (index = t*H*W + h*W + w). INTRA replaces attn1 only; attn2 (cross-attention to the UMT5
text tokens) stays dense. Layers follow a period-5 schedule

    i % 5 == 0  Scatter-A   group = (h % 2, w % 4) over all frames  (8 groups)
    i % 5 == 1  Gather      non-overlapping spatial tile over all frames
    i % 5 == 2  Scatter-B   group = (h % 4, w % 2) over all frames  (8 groups)
    i % 5 == 3  Gather      same tiles, torus-shifted by half a tile
    i % 5 == 4  Cone        1-D window of radius R = W_t*H*W + (H*W - 1) on the T-major sequence

and the first and last blocks (0 and 29) keep dense attention. Group ids depend only on (h, w), so
every Scatter/Gather group spans the full time axis; the Cone layer couples nearby frames at every
spatial position (its window covers the +/-W_t frame band, plus part of frame +/-(W_t+1)).

`sparse_self_attention` gathers each group with index_select and runs exact softmax attention on
it. It is differentiable and is what training and the VBench generations used. intra_wan_fast.py
computes the same pattern with FlashAttention-2 for inference speed.

Run `python intra_wan.py` for a CPU self-test against dense masked attention.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch


# ----------------------------------------------------------------------------------------------
# Configuration and layer schedule
# ----------------------------------------------------------------------------------------------
@dataclass
class INTRAConfig:
    scatter_a: tuple = (2, 4)        # (s_h, s_w) of the first Scatter layer in each period
    scatter_b: tuple = (4, 2)        # complementary strides for the second one
    tile: Optional[tuple] = None     # Gather tile (b_h, b_w); None = choose_tiles() per grid
    tile_target: int = 8             # auto tiles aim for about this many tiles per frame
    cone_wt: int = 4                 # Cone temporal half-band W_t (latent frames)
    period: int = 5


def layer_kind(layer_idx: int, cfg: INTRAConfig):
    """('scatter', (s_h, s_w)) | ('gather', shifted: bool) | ('cone', W_t)."""
    m = layer_idx % cfg.period
    if m == 0:
        return ("scatter", cfg.scatter_a)
    if m == 1:
        return ("gather", False)
    if m == 2:
        return ("scatter", cfg.scatter_b)
    if m == 3:
        return ("gather", True)
    return ("cone", cfg.cone_wt)


def choose_tiles(H: int, W: int, cfg: INTRAConfig):
    """Gather tile (b_h, b_w): divides (H, W), is at least the largest Scatter stride on each axis
    (so a tile always contains every Scatter residue class), and gives the tile count closest to
    cfg.tile_target."""
    if cfg.tile is not None:
        b_h, b_w = cfg.tile
        assert H % b_h == 0 and W % b_w == 0, f"tile {cfg.tile} must divide grid ({H},{W})"
        return b_h, b_w
    s_max_h = max(cfg.scatter_a[0], cfg.scatter_b[0])
    s_max_w = max(cfg.scatter_a[1], cfg.scatter_b[1])
    divs = lambda n: [d for d in range(1, n + 1) if n % d == 0]
    best, best_score = None, None
    for b_h in divs(H):
        if b_h < s_max_h:
            continue
        for b_w in divs(W):
            if b_w < s_max_w:
                continue
            score = abs((H // b_h) * (W // b_w) - cfg.tile_target)
            if best_score is None or score < best_score:
                best, best_score = (b_h, b_w), score
    assert best is not None, f"no valid tile for grid ({H},{W}) with strides >= ({s_max_h},{s_max_w})"
    return best


def cone_radius(W_t: int, frame: int):
    """Radius of the 1-D window on the T-major sequence that covers |dt| <= W_t at every position."""
    return W_t * frame + (frame - 1)


def grid_from_pixels(num_frames: int, height: int, width: int):
    """Token grid (T, H, W) for Wan2.2-TI2V-5B: VAE 4x temporal / 16x spatial, patch (1, 2, 2)."""
    return ((num_frames - 1) // 4 + 1, height // 32, width // 32)


# ----------------------------------------------------------------------------------------------
# Group construction (cached per grid)
# ----------------------------------------------------------------------------------------------
_CACHE = {}


def _coords(T, H, W, device):
    idx = torch.arange(T * H * W, device=device)
    r = idx % (H * W)
    return idx, idx // (H * W), r // W, r % W


def _groups_from_gid(idx, gid):
    return [idx[gid == g] for g in torch.unique(gid)]


def scatter_groups(T, H, W, s_h, s_w, device):
    key = ("scatter", T, H, W, s_h, s_w, str(device))
    if key not in _CACHE:
        idx, _, h, w = _coords(T, H, W, device)
        _CACHE[key] = _groups_from_gid(idx, (h % s_h) * s_w + (w % s_w))
    return _CACHE[key]


def gather_groups(T, H, W, b_h, b_w, off_h, off_w, device):
    key = ("gather", T, H, W, b_h, b_w, off_h, off_w, str(device))
    if key not in _CACHE:
        idx, _, h, w = _coords(T, H, W, device)
        n_tw = (W + b_w - 1) // b_w
        gid = (((h + off_h) % H) // b_h) * n_tw + ((w + off_w) % W) // b_w
        _CACHE[key] = _groups_from_gid(idx, gid)
    return _CACHE[key]


def layer_groups(cfg: INTRAConfig, grid, layer_idx: int, device):
    """Index groups of a Scatter/Gather layer (a partition of the grid); None for the Cone layer."""
    T, H, W = grid
    kind = layer_kind(layer_idx, cfg)
    if kind[0] == "scatter":
        return scatter_groups(T, H, W, *kind[1], device)
    if kind[0] == "gather":
        b_h, b_w = choose_tiles(H, W, cfg)
        off = (b_h // 2, b_w // 2) if kind[1] else (0, 0)
        return gather_groups(T, H, W, b_h, b_w, *off, device)
    return None


# ----------------------------------------------------------------------------------------------
# Reference attention. Tensors are (B, heads, N, D) with RoPE already applied.
# ----------------------------------------------------------------------------------------------
def _attn(q, k, v, scale):
    scores = torch.matmul(q, k.transpose(-2, -1)) * scale
    return torch.matmul(torch.softmax(scores, dim=-1), v)


def _dense_masked(q, k, v, scale, allow):
    scores = torch.matmul(q, k.transpose(-2, -1)) * scale
    scores = scores.masked_fill(~allow, float("-inf"))
    return torch.matmul(torch.softmax(scores, dim=-1), v)


def _grouped(q, k, v, scale, groups):
    out = torch.zeros_like(q)
    for idx in groups:
        out[:, :, idx] = _attn(q[:, :, idx], k[:, :, idx], v[:, :, idx], scale)
    return out


def sparse_self_attention(q, k, v, cfg: INTRAConfig, grid, layer_idx: int):
    """One INTRA attn1 layer. q, k, v: (B, heads, N, D)."""
    T, H, W = grid
    N = q.shape[2]
    assert N == T * H * W, f"token count {N} != T*H*W for grid {grid}"
    scale = q.shape[-1] ** -0.5
    groups = layer_groups(cfg, grid, layer_idx, q.device)
    if groups is not None:
        return _grouped(q, k, v, scale, groups)
    R = cone_radius(layer_kind(layer_idx, cfg)[1], H * W)
    i = torch.arange(N, device=q.device)
    return _dense_masked(q, k, v, scale, (i[:, None] - i[None, :]).abs() <= R)


# ----------------------------------------------------------------------------------------------
# diffusers 0.37 attn1 processor
# ----------------------------------------------------------------------------------------------
def apply_rotary_emb_037(hidden_states, freqs_cos, freqs_sin):
    """Copy of diffusers 0.37 transformer_wan.apply_rotary_emb, on the (B, N, heads, D) layout."""
    x1, x2 = hidden_states.unflatten(-1, (-1, 2)).unbind(-1)
    cos = freqs_cos[..., 0::2]
    sin = freqs_sin[..., 1::2]
    out = torch.empty_like(hidden_states)
    out[..., 0::2] = x1 * cos - x2 * sin
    out[..., 1::2] = x1 * sin + x2 * cos
    return out.type_as(hidden_states)


class WanINTRAAttnProcessor:
    """Replacement for diffusers 0.37 WanAttnProcessor on attn1. The q/k/v projections, QK-norm,
    RoPE and output projection mirror the stock processor; only the attention core changes.
    `attn_fn(q, k, v, cfg, grid, layer_idx)` takes and returns (B, heads, N, D)."""

    def __init__(self, layer_idx: int, cfg: INTRAConfig, attn_fn=sparse_self_attention):
        self.layer_idx = layer_idx
        self.cfg = cfg
        self.attn_fn = attn_fn
        self.grid = None  # (T, H, W), set with set_intra_grid before each forward

    def __call__(self, attn, hidden_states, encoder_hidden_states=None,
                 attention_mask=None, rotary_emb=None, **kwargs):
        assert encoder_hidden_states is None, "INTRA replaces attn1 (self-attention) only"
        assert self.grid is not None, "call set_intra_grid(transformer, grid) before forward"
        query = attn.norm_q(attn.to_q(hidden_states)).unflatten(2, (attn.heads, -1))
        key = attn.norm_k(attn.to_k(hidden_states)).unflatten(2, (attn.heads, -1))
        value = attn.to_v(hidden_states).unflatten(2, (attn.heads, -1))
        if rotary_emb is not None:
            query = apply_rotary_emb_037(query, *rotary_emb)
            key = apply_rotary_emb_037(key, *rotary_emb)
        out = self.attn_fn(query.transpose(1, 2), key.transpose(1, 2), value.transpose(1, 2),
                           self.cfg, self.grid, self.layer_idx)
        hs = out.transpose(1, 2).flatten(2, 3).type_as(query)
        return attn.to_out[1](attn.to_out[0](hs))


def install_intra_processors(transformer, cfg: INTRAConfig, dense_layers=None,
                             attn_fn=sparse_self_attention):
    """Install INTRA on attn1 of every block except `dense_layers` (default: first and last)."""
    n = len(transformer.blocks)
    dense_layers = {0, n - 1} if dense_layers is None else set(dense_layers)
    procs = {}
    for i, block in enumerate(transformer.blocks):
        if i not in dense_layers:
            procs[i] = WanINTRAAttnProcessor(i, cfg, attn_fn)
            block.attn1.set_processor(procs[i])
    return procs


def set_intra_grid(transformer, grid):
    for block in transformer.blocks:
        p = getattr(block.attn1, "processor", None)
        if isinstance(p, WanINTRAAttnProcessor):
            p.grid = tuple(grid)


def add_pattern_args(ap):
    ap.add_argument("--scatter_a", default="2,4", help="Scatter-A stride s_h,s_w")
    ap.add_argument("--scatter_b", default="4,2", help="Scatter-B stride s_h,s_w")
    ap.add_argument("--tile", default="auto", help="Gather tile b_h,b_w or 'auto'")
    ap.add_argument("--tile_target", type=int, default=8)
    ap.add_argument("--cone_wt", type=int, default=4, help="Cone temporal half-band (latent frames)")


def config_from_args(a) -> INTRAConfig:
    pair = lambda s: tuple(int(x) for x in s.split(","))
    return INTRAConfig(scatter_a=pair(a.scatter_a), scatter_b=pair(a.scatter_b),
                       tile=None if a.tile == "auto" else pair(a.tile),
                       tile_target=a.tile_target, cone_wt=a.cone_wt)


def add_dora_adapter(transformer, rank: int, ffn: bool = True):
    """DoRA on attn1 q/k/v/out (+ the FFN), alpha = rank. Shared by train.py and gen_vbench.py."""
    from peft import LoraConfig
    targets = ["attn1.to_q", "attn1.to_k", "attn1.to_v", "attn1.to_out.0"]
    if ffn:
        targets += ["ffn.net.0.proj", "ffn.net.2"]
    transformer.add_adapter(LoraConfig(r=rank, lora_alpha=rank, init_lora_weights="gaussian",
                                       use_dora=True, target_modules=targets))
    return targets


# ----------------------------------------------------------------------------------------------
# Self-test: grouped attention == dense masked attention (float64, CPU)
# ----------------------------------------------------------------------------------------------
def _allow_from_groups(groups, N):
    gid = torch.empty(N, dtype=torch.long)
    for g, idx in enumerate(groups):
        gid[idx] = g
    return gid[:, None] == gid[None, :]


def _selftest():
    torch.manual_seed(0)
    T, H, W, Hd, D = 4, 8, 8, 2, 8
    N, grid, frame = T * H * W, (T, H, W), H * W
    cfg = INTRAConfig(tile=(4, 4), cone_wt=1)
    q, k, v = (torch.randn(1, Hd, N, D, dtype=torch.float64) for _ in range(3))
    scale = D ** -0.5
    _, t, h, w = _coords(T, H, W, "cpu")
    ok = True

    def check(name, got, allow):
        nonlocal ok
        err = (got - _dense_masked(q, k, v, scale, allow)).abs().max().item()
        ok &= err < 1e-9
        print(f"  [{'OK  ' if err < 1e-9 else 'FAIL'}] {name:24s} max err {err:.1e}")

    print(f"grid {grid}, N={N}, auto tile for (15,26): {choose_tiles(15, 26, INTRAConfig())}")
    for li in range(cfg.period):
        kind = layer_kind(li, cfg)
        got = sparse_self_attention(q, k, v, cfg, grid, li)
        if kind[0] == "scatter":
            s_h, s_w = kind[1]
            hh, ww = h % s_h, w % s_w
            allow = (hh[:, None] == hh[None, :]) & (ww[:, None] == ww[None, :])
        elif kind[0] == "gather":
            b_h, b_w = choose_tiles(H, W, cfg)
            o_h, o_w = (b_h // 2, b_w // 2) if kind[1] else (0, 0)
            th, tw = ((h + o_h) % H) // b_h, ((w + o_w) % W) // b_w
            allow = (th[:, None] == th[None, :]) & (tw[:, None] == tw[None, :])
        else:
            i = torch.arange(N)
            allow = (i[:, None] - i[None, :]).abs() <= cone_radius(kind[1], frame)
            dt = (t[:, None] - t[None, :]).abs()
            contained = bool(((dt <= kind[1]) & ~allow).sum() == 0) and \
                bool((allow & ~(dt <= kind[1] + 1)).sum() == 0)
            ok &= contained
            print(f"  [{'OK  ' if contained else 'FAIL'}] cone band |dt|<=W_t  is inside window inside |dt|<=W_t+1")
        groups = layer_groups(cfg, grid, li, "cpu")
        if groups is not None:   # every Scatter/Gather layer must partition the grid
            part = torch.cat(groups).sort().values.equal(torch.arange(N))
            same = _allow_from_groups(groups, N).equal(allow)
            ok &= part and same
            print(f"  [{'OK  ' if part and same else 'FAIL'}] L{li} groups partition grid, match mask")
        check(f"L{li} {kind[0]}{'-shift' if kind == ('gather', True) else ''}", got, allow)
    print("SELF-TEST:", "PASS" if ok else "FAIL")
    return ok


if __name__ == "__main__":
    import sys
    sys.exit(0 if _selftest() else 1)
