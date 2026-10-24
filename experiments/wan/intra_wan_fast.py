"""
INTRA attn1 for Wan2.2 on FlashAttention-2 (inference / speed path, CUDA only).

Same attention pattern as intra_wan.sparse_self_attention (the groups come from the same builders);
only the computation differs:

  * Scatter / Gather layers are block-diagonal once tokens are laid out group by group: permute
    q, k, v into group-contiguous order, run one varlen FlashAttention-2 call with cu_seqlens = group
    sizes, and invert the permutation. The two gathers are included in every timed call.
  * The Cone layer is a 1-D window of radius R on the natural T-major order, i.e. FlashAttention-2
    with window_size = (R, R). No permutation.

Everything goes through torch.ops.aten._flash_attention_forward (the FlashAttention-2 build bundled
with PyTorch), the kernel that also serves dense F.scaled_dot_product_attention on A100, so the
dense baseline and INTRA use the same kernel family.
"""
from __future__ import annotations

import dataclasses

import torch

from intra_wan import (INTRAConfig, layer_kind, layer_groups, cone_radius,
                       install_intra_processors, set_intra_grid)

_fa = torch.ops.aten._flash_attention_forward.default
_PLANS = {}


def fa2_dense(q, k, v, scale, window=None):
    """q, k, v: (B, N, H, D) contiguous -> (B, N, H, D). `window` = symmetric radius or None."""
    N = q.shape[1]
    w = None if window is None else int(window)
    return _fa(q, k, v, None, None, N, N, 0.0, False, False, scale=scale,
               window_size_left=w, window_size_right=w)[0]


def fa2_varlen(qf, kf, vf, cu, max_len, scale):
    """qf, kf, vf: (total, H, D) packed by cu_seqlens `cu` -> (total, H, D)."""
    return _fa(qf, kf, vf, cu, cu, max_len, max_len, 0.0, False, False, scale=scale)[0]


def _build_plan(cfg: INTRAConfig, grid, layer_idx, B, dev):
    T, H, W = grid
    N = T * H * W
    groups = layer_groups(cfg, grid, layer_idx, dev)
    if groups is None:                                   # Cone
        R = cone_radius(layer_kind(layer_idx, cfg)[1], H * W)
        i = torch.arange(N, dtype=torch.int64)
        pairs = int((torch.clamp(i + R, max=N - 1) - torch.clamp(i - R, min=0) + 1).sum())
        return ("window", R, pairs)
    perm1 = torch.cat(groups).to(dev)
    assert perm1.numel() == N and torch.unique(perm1).numel() == N, "groups must partition the grid"
    perm = torch.cat([b * N + perm1 for b in range(B)])
    inv = torch.empty_like(perm)
    inv[perm] = torch.arange(perm.numel(), device=dev)
    lens = [g.numel() for g in groups]
    cu = torch.zeros(len(lens) * B + 1, dtype=torch.int32, device=dev)
    cu[1:] = torch.tensor(lens * B, dtype=torch.int32, device=dev).cumsum(0)
    return ("group", perm, inv, cu, max(lens), sum(l * l for l in lens))


def layer_plan(cfg: INTRAConfig, grid, layer_idx, B, dev):
    """('group', perm, inv, cu_seqlens, max_len, pairs) | ('window', R, pairs), cached.
    `pairs` = attended (q, k) pairs per batch element and head (for density reporting)."""
    key = (dataclasses.astuple(cfg), tuple(grid), layer_idx, B, str(dev))
    if key not in _PLANS:
        _PLANS[key] = _build_plan(cfg, grid, layer_idx, B, dev)
    return _PLANS[key]


def fast_sparse_self_attention(q, k, v, cfg: INTRAConfig, grid, layer_idx: int):
    """Drop-in for intra_wan.sparse_self_attention (forward only). q, k, v: (B, heads, N, D); the
    processor passes transposed views of contiguous (B, N, heads, D) tensors."""
    B, Hd, N, D = q.shape
    scale = D ** -0.5
    qn, kn, vn = (x.transpose(1, 2) for x in (q, k, v))          # (B, N, heads, D)
    plan = layer_plan(cfg, grid, layer_idx, B, q.device)
    if plan[0] == "window":
        o = fa2_dense(qn.contiguous(), kn.contiguous(), vn.contiguous(), scale, window=plan[1])
        return o.transpose(1, 2)
    _, perm, inv, cu, max_len, _ = plan
    flat = lambda x: x.reshape(B * N, Hd, D).index_select(0, perm)
    o = fa2_varlen(flat(qn), flat(kn), flat(vn), cu, max_len, scale)
    return o.index_select(0, inv).view(B, N, Hd, D).transpose(1, 2)


def install_fast_intra(transformer, cfg: INTRAConfig, grid, dense_layers=None):
    procs = install_intra_processors(transformer, cfg, dense_layers,
                                     attn_fn=fast_sparse_self_attention)
    set_intra_grid(transformer, grid)
    return procs
