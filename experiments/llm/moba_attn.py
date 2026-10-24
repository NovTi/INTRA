#!/usr/bin/env python3
"""MoBA (Mixture of Block Attention, Lu et al. 2025) as a transformers attention backend,
memory-bounded.

The gate logic is a line-by-line port of upstream `moba_attn_varlen_naive`
(https://github.com/MoonshotAI/MoBA, moba/moba_naive.py), evaluated in query chunks so peak memory
is O(H * q_block * S) instead of the O(H * S * S) float32 gate/score tensors the naive reference
materialises. It needs no flash_attn. verify_against_reference() checks it element-wise against
the upstream function; run this file directly with MOBA_REPO pointing at a clone of the MoBA repo.

Selected blocks are applied as an additive mask to SDPA, so this does DENSE FLOPs: it is faithful
in quality, not in speed. MoBA is prefill-only in the upstream wrapper (decode falls through to
dense), and that is preserved: q_len == 1 goes to SDPA.
"""
import math
import torch
import torch.nn.functional as F


def _moba_gate(q_chunk, k_block_mean, q_pos, chunk_size, topk, density=None):
    """Boolean [H, C, N] -- which KV blocks each query in this chunk attends to.

    Ported from moba_attn_varlen_naive lines 51-75. Three rules, in order:
      1. a query may not see a block that starts after it              -> -inf
      2. a query always sees its own block                             -> +inf
      3. of the rest, keep the top-k by dot product with the block mean
    Rule 2 must be applied after rule 1 because the reference sets the whole prefix to -inf first and
    then overwrites the diagonal band; doing it the other way round loses the forced self-block.
    """
    H, C, D = q_chunk.shape[0], q_chunk.shape[1], q_chunk.shape[2]
    N = k_block_mean.shape[0]

    gate = torch.einsum("hcd,nhd->hcn", q_chunk.float(), k_block_mean.float())  # [H, C, N]

    bidx = torch.arange(N, device=q_chunk.device).view(1, 1, N)
    pos = q_pos.view(1, C, 1)
    own = (pos // chunk_size) == bidx                      # query sits inside this block
    future = (pos < (bidx + 1) * chunk_size) & (~own)      # block begins at or after the query
    gate = gate.masked_fill(future, float("-inf"))
    gate = gate.masked_fill(own, float("inf"))

    kk = resolve_topk(N, topk, density)
    top_val, top_idx = torch.topk(gate, k=kk, dim=-1, largest=True, sorted=False)
    thresh = top_val.min(dim=-1).values.unsqueeze(-1)
    need = gate >= thresh
    # guard against ties selecting more than k blocks, exactly as the reference does
    idx_mask = torch.zeros_like(need)
    idx_mask.scatter_(dim=-1, index=top_idx, value=True)
    return torch.logical_and(need, idx_mask)


def resolve_topk(n_blocks, topk=None, density=None):
    """Blocks to keep. Either a fixed count, or a fraction of the sequence.

    A fixed top-k does not hold density constant (chunk 2048 / top-3 keeps 37.5% of a 16K prompt
    but only 11% of a 54K one); `density` keeps a fixed fraction of the causal blocks instead.
    """
    if density is not None:
        return max(1, min(n_blocks, round(density * n_blocks)))
    return max(1, min(n_blocks, topk if topk is not None else 3))


def moba_attn(q, k, v, chunk_size=2048, topk=3, density=None, q_block=None, mask_dtype=None):
    """q: [B, Hq, S, D], k/v: [B, Hkv, S, D] (HF layout). Returns [B, S, Hq, D].

    Assumes prefill: S_q == S_k, causal, no padding. GQA is expanded here because MoBA's gate is
    per-query-head and the upstream wrapper expands too (wrapper.py:59-61).
    """
    B, Hq, S, D = q.shape
    Hkv = k.shape[1]
    if Hkv != Hq:
        rep = Hq // Hkv
        k = k.repeat_interleave(rep, dim=1)
        v = v.repeat_interleave(rep, dim=1)

    mask_dtype = mask_dtype or q.dtype
    if q_block is None:
        # keep the additive mask under ~1 GB: Hq * q_block * S * itemsize
        itemsize = torch.finfo(mask_dtype).bits // 8
        q_block = max(64, min(2048, int(1 << 30) // max(1, Hq * S * itemsize)))

    N = math.ceil(S / chunk_size)
    scale = D ** -0.5
    out = torch.empty(B, Hq, S, D, dtype=q.dtype, device=q.device)

    for b in range(B):
        kb = k[b]                                            # [H, S, D]
        # block means over the key axis -> [N, H, D] to match the reference's layout
        pad = N * chunk_size - S
        km = F.pad(kb, (0, 0, 0, pad)).view(Hq, N, chunk_size, D)
        if pad:
            cnt = torch.full((N,), float(chunk_size), device=q.device, dtype=torch.float32)
            cnt[-1] = chunk_size - pad
            k_block_mean = km.float().sum(dim=2) / cnt.view(1, N, 1)
        else:
            k_block_mean = km.float().mean(dim=2)
        k_block_mean = k_block_mean.permute(1, 0, 2).contiguous()   # [N, H, D]

        for qs in range(0, S, q_block):
            qe = min(S, qs + q_block)
            q_pos = torch.arange(qs, qe, device=q.device)
            qc = q[b, :, qs:qe, :]                                   # [H, C, D]
            need = _moba_gate(qc, k_block_mean, q_pos, chunk_size, topk, density)  # [H, C, N]

            # block selection -> per-key additive mask, then causal within the selected blocks
            keep = need.repeat_interleave(chunk_size, dim=-1)[:, :, :S]    # [H, C, S]
            causal = q_pos.view(1, -1, 1) >= torch.arange(S, device=q.device).view(1, 1, -1)
            keep = keep & causal
            add = torch.zeros(keep.shape, dtype=mask_dtype, device=q.device)
            add.masked_fill_(~keep, torch.finfo(mask_dtype).min)

            o = F.scaled_dot_product_attention(
                qc.unsqueeze(0), kb.unsqueeze(0), v[b].unsqueeze(0),
                attn_mask=add.unsqueeze(0), scale=scale,
            )
            out[b, :, qs:qe, :] = o.squeeze(0)
            del keep, add, need

    return out.transpose(1, 2).contiguous()          # [B, S, Hq, D]


# ---------------------------------------------------------------- transformers integration

def make_moba_attention(chunk_size=2048, topk=3, density=None):
    """Returns a callable matching the transformers ALL_ATTENTION_FUNCTIONS contract."""
    def moba_attention_forward(module, query, key, value, attention_mask=None,
                               scaling=None, dropout=0.0, **kwargs):
        # decode: one query token, nothing to select over -- upstream falls back to dense here too
        if query.shape[2] == 1:
            n_rep = query.shape[1] // key.shape[1]
            k2 = key.repeat_interleave(n_rep, dim=1) if n_rep > 1 else key
            v2 = value.repeat_interleave(n_rep, dim=1) if n_rep > 1 else value
            am = attention_mask
            if am is not None:
                am = am[:, :, :, : k2.shape[-2]]
            o = F.scaled_dot_product_attention(query, k2, v2, attn_mask=am,
                                               scale=scaling, is_causal=False)
            return o.transpose(1, 2).contiguous(), None
        return moba_attn(query, key, value, chunk_size=chunk_size, topk=topk,
                         density=density), None
    return moba_attention_forward


# ---------------------------------------------------------------- correctness

def verify_against_reference(S=512, H=4, D=64, chunk_size=64, topk=3, device="cuda", seed=0):
    """Element-wise check against the upstream educational implementation."""
    # Load moba_naive.py by path: importing the `moba` package would pull in flash_attn via
    # moba_efficient.py, while the reference function itself only needs torch.
    import os, importlib.util
    ref_path = os.path.join(os.environ.get("MOBA_REPO", "MoBA"), "moba", "moba_naive.py")
    spec = importlib.util.spec_from_file_location("_moba_naive_ref", ref_path)
    _ref_mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(_ref_mod)
    moba_attn_varlen_naive = _ref_mod.moba_attn_varlen_naive

    torch.manual_seed(seed)
    dt = torch.float32   # reference upcasts internally; compare in fp32 to isolate algorithm from precision
    q = torch.randn(S, H, D, device=device, dtype=dt)
    k = torch.randn(S, H, D, device=device, dtype=dt)
    v = torch.randn(S, H, D, device=device, dtype=dt)
    cu = torch.tensor([0, S], device=device, dtype=torch.int32)

    ref = moba_attn_varlen_naive(q, k, v, cu, S, chunk_size, topk)        # [S, H, D]
    mine = moba_attn(q.permute(1, 0, 2).unsqueeze(0),                     # -> [1, H, S, D]
                     k.permute(1, 0, 2).unsqueeze(0),
                     v.permute(1, 0, 2).unsqueeze(0),
                     chunk_size=chunk_size, topk=topk, q_block=128,
                     mask_dtype=torch.float32).squeeze(0)                 # -> [S, H, D]

    diff = (ref - mine).abs()
    rel = diff.max().item() / max(1e-9, ref.abs().max().item())
    print(f"  S={S} H={H} D={D} chunk={chunk_size} topk={topk}: "
          f"max|diff|={diff.max().item():.3e}  rel={rel:.3e}  mean|diff|={diff.mean().item():.3e}")
    return rel


if __name__ == "__main__":
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"verifying chunked MoBA against moba/moba_naive.py on {dev}")
    worst = 0.0
    for S, H, D, c, t in [(512, 4, 64, 64, 3), (512, 4, 64, 128, 2), (1024, 8, 64, 128, 4),
                          (768, 4, 128, 256, 1), (1000, 4, 64, 64, 3)]:   # last: S not divisible by chunk
        worst = max(worst, verify_against_reference(S, H, D, c, t, device=dev))
    print(f"\nworst relative error {worst:.3e} -> {'PASS' if worst < 1e-4 else 'FAIL'}")
