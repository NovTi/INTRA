#!/usr/bin/env python3
"""NSA (Native Sparse Attention, Yuan et al. 2025) as a post-hoc sparse pattern for a pretrained model.

MoBA is parameter-free and ports over verbatim; NSA does not, and the two places where it cannot be
transferred to an already-pretrained model are stated below.

WHAT NSA IS: for each query, three complementary key sets, whose outputs are summed through a
learned gate.
  1. COMPRESSION  keys/values pooled into overlapping blocks (block l, stride d) -> a coarse view of
                  the ENTIRE history at 1/d the cost.
  2. SELECTION    the top-n blocks (block l') by importance, attended at FULL resolution.
  3. WINDOW       the last w keys, at full resolution.
It is the compression branch that distinguishes NSA from MoBA: MoBA selects blocks and sees nothing
else, so anything outside its top-k is invisible, while NSA always retains a lossy view of all of it.
That is the comparison worth running, and it is preserved exactly here.

TWO DEVIATIONS, both forced by applying a PRETRAINING-NATIVE architecture to a frozen model:

  (a) COMPRESSION IS MEAN POOLING, not NSA's learned MLP-with-intra-block-positional-encoding. That
      MLP is trained from scratch during pretraining. There is no way to obtain it for an
      already-pretrained LLaMA short of pretraining one, and a randomly-initialised MLP in the
      compression path would measure the random init, not NSA. Mean pooling is the parameter-free
      stand-in, and it is the same pooling MoBA uses for its gate -- so the two baselines are
      handicapped identically rather than one of them getting a learned component.

  (b) THE THREE BRANCHES ARE COMBINED BY UNION UNDER ONE SOFTMAX, not by NSA's learned per-query
      gate over three separately-normalised outputs. Same reason: the gate MLP is pretrained. With a
      FIXED gate, summing three independently-normalised attention outputs does not produce a convex
      combination of value vectors at all, and the model is far off-distribution before any LoRA sees
      it -- that would measure the arbitrary gate constant. Taking the union of the three key sets
      under a single softmax keeps the output a proper attention distribution, is exactly how INTRA
      and MoBA are evaluated here, and changes NOTHING about which keys are visible, which is the
      property under test. (The `combine` argument is accepted but only "union" is implemented.)

Cost model, per query at position i (context i):
    compression   ceil(i/d) pooled keys        -> a 1/d fraction, present at every length
    selection     n * l'    full keys
    window        w         full keys
`density=` sets n to hit a target TOTAL fraction of i, which is what makes an iso-density comparison
against INTRA's fixed 1/8 Scatter and MoBA's --moba_density possible at all. SCROLLS inputs run 2K
to 54K tokens, so a fixed n is a different fraction at every length and would confound method with
length.

Like moba_attn.py this materialises an additive mask and therefore does DENSE FLOPs. It is faithful
in QUALITY, not in speed; no timing claim may be made from it.
"""
import math

import torch
import torch.nn.functional as F


def compress_kv(k, v, block, stride):
    """Mean-pool k/v into overlapping blocks. Returns (kc, vc, end_pos).

    kc/vc: [B,H,Nc,D].  end_pos[j] is the LAST original position inside compressed block j, which is
    what causality has to be tested against -- a query may use a pooled block only once the whole
    block is in its past, otherwise the pooled vector leaks future tokens into the average. Getting
    this wrong is silent: the model simply scores better than it should.

    Vectorised with unfold rather than a Python loop over block starts (same result, far fewer
    kernel launches at long sequence lengths).
    """
    B, H, S, D = k.shape
    block = min(block, S)
    n_full = max(0, (S - block) // stride + 1)
    parts_k, parts_v, starts = [], [], []
    if n_full > 0:
        # unfold -> [B,H,n_full,D,block]; mean over the block axis
        parts_k.append(k.unfold(2, block, stride).mean(dim=-1))
        parts_v.append(v.unfold(2, block, stride).mean(dim=-1))
        starts.extend(range(0, n_full * stride, stride))
    last_start = (n_full - 1) * stride if n_full > 0 else -1
    if last_start + block < S:
        # tail block, so the final positions are not invisible to the compression branch
        s0 = max(0, S - block)
        parts_k.append(k[:, :, s0:].mean(dim=2, keepdim=True))
        parts_v.append(v[:, :, s0:].mean(dim=2, keepdim=True))
        starts.append(s0)
    kc = torch.cat(parts_k, dim=2)
    vc = torch.cat(parts_v, dim=2)
    end_pos = torch.clamp(torch.tensor(starts, device=k.device) + block - 1, max=S - 1)
    return kc, vc, end_pos


def plan_budget(S, density, sel_block=64, f_win=0.4, f_cmp=0.2):
    """Size all THREE branches to a total density target. Returns (window, cmp_block, cmp_stride, kk).

    NSA's published hyperparameters are absolute (w=512, l=32, d=16, n=16), so its density is
    whatever those imply at a given length. At S=4096 the window alone is 512 keys and compression
    adds 255 pooled ones, which already exceed a 0.125*4096 budget, so a low density target is
    unreachable while holding them fixed. The budget is therefore split across the branches in a
    fixed proportion (40% window / 20% compression / 40% selection), so NSA keeps its three-branch
    character at every density.

    Budget is sized against the MEAN causal context S/2, because density here is measured against
    the causal key budget -- the same denominator INTRA's 1/8 Scatter and MoBA's --moba_density use.
    """
    mean_ctx = max(1.0, S / 2.0)
    total = density * mean_ctx
    window = max(64, int(f_win * total))
    n_cmp_target = max(1.0, f_cmp * total)
    cmp_stride = max(1, int(S / n_cmp_target))
    cmp_block = max(2, 2 * cmp_stride)                     # NSA uses l = 2d (32/16)

    # OVERLAP CORRECTION. The branches are a UNION, not a sum: the query's own forced selection block
    # sits inside the sliding window whenever window > sel_block, so one sel_block of the budget is
    # double-counted. That waste is a fixed absolute size while the budget grows with S, so leaving
    # it uncorrected makes density climb with length (0.099 / 0.126 / 0.153 at S = 2048 / 4096 /
    # 8192). Adding it back, and rounding rather than flooring, keeps density flat.
    overlap = min(window, sel_block)
    kk = max(1, round((total - window - n_cmp_target + overlap) / sel_block))
    return window, cmp_block, cmp_stride, kk


def resolve_n_sel(S, sel_block, n_sel=None, density=None, window=0, n_cmp=0, cmp_stride=1):
    """Selection-branch block count when the branch sizes are given absolutely (density is None)."""
    n_blocks = math.ceil(S / sel_block)
    if density is None:
        return max(1, min(n_blocks, n_sel if n_sel is not None else 16))
    budget = density * (S / 2.0) - window - n_cmp
    return max(1, min(n_blocks, int(budget // sel_block)))


def nsa_attn(q, k, v, cmp_block=32, cmp_stride=16, sel_block=64, n_sel=16, density=None,
             window=512, use_cmp=True, combine="union", q_block=None, mask_dtype=None,
             budget_split=True, stats=None):
    """q: [B,Hq,S,D], k/v: [B,Hkv,S,D] (HF layout). Returns [B,S,Hq,D].

    Prefill only: S_q == S_k, causal, no padding. GQA is expanded because selection is per query
    head, exactly as in moba_attn.

    stats: optional dict; if given, "kept"/"total" accumulate the mask's true occupancy so density
    is MEASURED off the mask that actually ran rather than recomputed by a second implementation
    that could drift away from it.
    """
    B, Hq, S, D = q.shape
    Hkv = k.shape[1]
    if Hkv != Hq:
        rep = Hq // Hkv
        k = k.repeat_interleave(rep, dim=1)
        v = v.repeat_interleave(rep, dim=1)

    mask_dtype = mask_dtype or q.dtype
    neg = torch.finfo(mask_dtype).min
    scale = D ** -0.5
    dev = q.device

    # With a density target, ALL THREE branches are resized to fit it (see plan_budget). Holding the
    # published absolute window/stride and charging the shortfall to selection cannot reach low
    # densities at SCROLLS lengths.
    if density is not None and budget_split:
        window, cmp_block, cmp_stride, n_sel = plan_budget(S, density, sel_block)
        density = None                      # branches are now sized; do not re-derive from density

    kc = vc = None
    n_cmp = 0
    if use_cmp and cmp_block > 0:
        kc, vc, cmp_end = compress_kv(k, v, cmp_block, cmp_stride)
        n_cmp = kc.shape[2]

    n_blocks = math.ceil(S / sel_block)
    kk = resolve_n_sel(S, sel_block, n_sel, density, window, n_cmp, cmp_stride)

    # block means for the SELECTION score. NSA reuses the compression branch's attention scores here
    # as a speed optimisation (it lets one set of scores drive both branches); recomputing from a
    # separate pooling is equivalent in what it selects and keeps sel_block independent of cmp_block.
    pad = n_blocks * sel_block - S
    ksel = F.pad(k, (0, 0, 0, pad)).view(B, Hq, n_blocks, sel_block, D)
    if pad:
        cnt = torch.full((n_blocks,), float(sel_block), device=dev, dtype=torch.float32)
        cnt[-1] = sel_block - pad
        k_blk = ksel.float().sum(dim=3) / cnt.view(1, 1, n_blocks, 1)
    else:
        k_blk = ksel.float().mean(dim=3)                       # [B,H,N,D]
    del ksel

    if q_block is None:
        itemsize = torch.finfo(mask_dtype).bits // 8
        q_block = max(64, min(2048, int(1 << 30) // max(1, Hq * (S + n_cmp) * itemsize)))

    out = torch.empty(B, Hq, S, D, dtype=q.dtype, device=q.device)
    key_pos = torch.arange(S, device=dev)
    blk_idx = torch.arange(n_blocks, device=dev)

    for b in range(B):
        k_all = torch.cat([kc[b], k[b]], dim=1) if n_cmp else k[b]     # [H, n_cmp+S, D]
        v_all = torch.cat([vc[b], v[b]], dim=1) if n_cmp else v[b]
        for qs in range(0, S, q_block):
            qe = min(S, qs + q_block)
            qpos = torch.arange(qs, qe, device=dev)
            qc = q[b, :, qs:qe, :]                                     # [H,C,D]

            # ---- selection: top-kk blocks by q . blockmean, own block forced, future forbidden
            score = torch.einsum("hcd,hnd->hcn", qc.float(), k_blk[b])          # [H,C,N]
            own = (qpos.view(1, -1, 1) // sel_block) == blk_idx.view(1, 1, -1)
            future = (qpos.view(1, -1, 1) < blk_idx.view(1, 1, -1) * sel_block) & (~own)
            score = score.masked_fill(future, float("-inf")).masked_fill(own, float("inf"))
            top_idx = torch.topk(score, k=kk, dim=-1, largest=True, sorted=False).indices
            sel = torch.zeros_like(score, dtype=torch.bool).scatter_(-1, top_idx, True)
            del score, top_idx

            keep = sel.repeat_interleave(sel_block, dim=-1)[:, :, :S]           # [H,C,S]
            del sel
            # ---- window branch: union'd in, not a separate softmax
            if window > 0:
                inwin = (key_pos.view(1, 1, -1) > (qpos.view(1, -1, 1) - window))
                keep = keep | inwin
            keep = keep & (key_pos.view(1, 1, -1) <= qpos.view(1, -1, 1))       # causal

            if n_cmp:
                # a pooled block is usable only once ALL of its source positions are in the past
                okc = (cmp_end.view(1, 1, -1) <= qpos.view(1, -1, 1)).expand(keep.shape[0], -1, -1)
                keep = torch.cat([okc, keep], dim=-1)

            if stats is not None:
                # denominator is the DENSE causal budget for these rows, so the ratio is directly
                # comparable to INTRA's 0.126 and MoBA's 0.125
                stats["kept"] = stats.get("kept", 0) + int(keep[0].sum().item())
                stats["total"] = stats.get("total", 0) + int((qpos + 1).sum().item())

            add = torch.zeros(keep.shape, dtype=mask_dtype, device=dev)
            add.masked_fill_(~keep, neg)
            # every query keeps its own block and its own position, so no row can be all -inf
            o = F.scaled_dot_product_attention(qc.unsqueeze(0), k_all.unsqueeze(0),
                                               v_all.unsqueeze(0), attn_mask=add.unsqueeze(0),
                                               scale=scale)
            out[b, :, qs:qe, :] = o.squeeze(0)
            del keep, add

    return out.transpose(1, 2).contiguous()


def make_nsa_attention(**cfg):
    """Callable matching the transformers ALL_ATTENTION_FUNCTIONS contract.

    nsa_attn is a prefill kernel (it derives every block count from q.shape[2], assuming
    S_q == S_k), so single-token decode steps fall back to dense attention over the KV cache.
    """
    def nsa_attention_forward(module, query, key, value, attention_mask=None,
                              scaling=None, dropout=0.0, **kwargs):
        if query.shape[2] == 1 and key.shape[2] > 1:
            # is_causal MUST be False: with q_len=1 and kv_len=N, SDPA aligns the causal mask to the
            # TOP-LEFT, which would leave the single query attending to key 0 alone.
            sc = scaling if scaling is not None else getattr(
                module, "scaling", query.shape[-1] ** -0.5)
            Hq, Hkv = query.shape[1], key.shape[1]
            k2, v2 = key, value
            if Hq != Hkv:
                rep = Hq // Hkv
                k2 = key.repeat_interleave(rep, dim=1)
                v2 = value.repeat_interleave(rep, dim=1)
            am = attention_mask[:, :, :, : k2.shape[-2]] if attention_mask is not None else None
            o = F.scaled_dot_product_attention(query, k2, v2, attn_mask=am, scale=sc,
                                               is_causal=False)
            return o.transpose(1, 2).contiguous(), None
        if scaling is None:
            scaling = getattr(module, "scaling", query.shape[-1] ** -0.5)
        return nsa_attn(query, key, value, **cfg), None
    return nsa_attention_forward


# ---------------------------------------------------------------- self-test
def _dense(q, k, v):
    Hq, Hkv = q.shape[1], k.shape[1]
    if Hq != Hkv:
        rep = Hq // Hkv
        k = k.repeat_interleave(rep, dim=1); v = v.repeat_interleave(rep, dim=1)
    return F.scaled_dot_product_attention(q, k, v, is_causal=True,
                                          scale=q.shape[-1] ** -0.5).transpose(1, 2).contiguous()


def measured_density(q, k, v, **cfg):
    """Density of the mask that actually ran. One source of truth -- see nsa_attn(stats=)."""
    st = {}
    nsa_attn(q, k, v, stats=st, **cfg)
    return st["kept"] / max(1, st["total"])


def calibrate_density(q, k, v, target, lo=0.01, hi=2.0, iters=12, **cfg):
    """Binary-search the NOMINAL density so the MEASURED one lands on `target`.

    The nominal budget systematically overshoots the realised mask because the branches OVERLAP: the
    sliding window and the query's own forced selection block cover much of the same ground, so a
    union of "102 window + 2x64 selected + 51 pooled" keys is nowhere near 281 distinct keys. The
    gap is large -- nominal 0.125 measures 0.061 -- and it is not a constant factor, so it has to be
    solved rather than assumed.

    Searching against measured_density means the answer is defined by the mask that actually runs.
    """
    for _ in range(iters):
        mid = (lo + hi) / 2
        if measured_density(q, k, v, density=mid, **cfg) < target:
            lo = mid
        else:
            hi = mid
    nom = (lo + hi) / 2
    return nom, measured_density(q, k, v, density=nom, **cfg)


if __name__ == "__main__":
    torch.manual_seed(0)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    B, Hq, Hkv, S, D = 1, 8, 2, 512, 64
    q = torch.randn(B, Hq, S, D, device=dev)
    k = torch.randn(B, Hkv, S, D, device=dev)
    v = torch.randn(B, Hkv, S, D, device=dev)

    # 1. REDUCES TO DENSE. With the window covering the whole context and no compression branch,
    #    every key is in the union, so the output must equal plain causal SDPA. This is the test that
    #    catches an off-by-one in the causal or window masks -- a mask bug that only *loses* keys
    #    would otherwise look like "sparse attention working as intended".
    o_dense = _dense(q, k, v)
    o_full = nsa_attn(q, k, v, window=S, n_sel=1, use_cmp=False, sel_block=64)
    err = (o_full - o_dense).abs().max().item() / o_dense.abs().max().item()
    print(f"  reduces-to-dense: rel err {err:.3e}")
    assert err < 1e-5, f"NSA with a full window is not dense (rel err {err:.3e})"

    # 2. CAUSALITY. Perturbing a key strictly after position i must not change output i. The pooled
    #    compression branch is the easy place to leak the future, so this runs WITH it enabled.
    o1 = nsa_attn(q, k, v, window=64, n_sel=2, use_cmp=True, cmp_block=32, cmp_stride=16)
    k2 = k.clone(); k2[:, :, S - 1] += 10.0
    v2 = v.clone(); v2[:, :, S - 1] += 10.0
    o2 = nsa_attn(q, k2, v2, window=64, n_sel=2, use_cmp=True, cmp_block=32, cmp_stride=16)
    leak = (o1[:, : S - 1] - o2[:, : S - 1]).abs().max().item()
    print(f"  causality (last key perturbed): max change to earlier outputs {leak:.3e}")
    assert leak < 1e-4, f"future leaked into the past: {leak:.3e}"

    # 3. DENSITY, measured off the mask that ran. S=4096 is the shortest length at which window,
    #    selection and compression are all genuinely partial (at S=512 a 512-wide window is dense).
    SL = 4096
    ql = torch.randn(1, 2, SL, D, device=dev)
    kl = torch.randn(1, 2, SL, D, device=dev)
    vl = torch.randn(1, 2, SL, D, device=dev)
    print(f"  {'config (S=4096)':<58}{'density':>9}")
    for cfg in [dict(cmp_block=32, cmp_stride=16, sel_block=64, n_sel=16, window=512),
                dict(cmp_block=32, cmp_stride=16, sel_block=64, density=0.125, window=512),
                dict(cmp_block=32, cmp_stride=16, sel_block=64, density=0.25, window=512),
                dict(use_cmp=False, sel_block=64, density=0.125, window=512)]:
        d = measured_density(ql, kl, vl, **cfg)
        desc = " ".join(f"{a}={b}" for a, b in cfg.items())
        print(f"  {desc:<58}{d:>9.3f}")
    # the density knob must actually move density, or --nsa_density is decorative
    d_lo = measured_density(ql, kl, vl, sel_block=64, density=0.125)
    d_hi = measured_density(ql, kl, vl, sel_block=64, density=0.25)
    assert d_hi > d_lo + 0.05, f"--nsa_density does not move density ({d_lo:.3f} -> {d_hi:.3f})"
    assert d_lo < 0.5, f"density=0.125 produced {d_lo:.3f}; it is not sparse at all"

    # 4. CALIBRATION + LENGTH-INVARIANCE: the calibrated nominal density must produce a measured
    #    density that barely moves with length (plan_budget scales every branch with S).
    TARGET = 0.126
    nom, got = calibrate_density(ql, kl, vl, TARGET, sel_block=64)
    print(f"  calibrated nominal density {nom:.4f} -> measured {got:.4f} (target {TARGET})")
    assert abs(got - TARGET) < 0.01, f"calibration missed: {got:.4f}"
    print(f"  {'length':<10}{'measured density at that nominal':>34}")
    for SS in (2048, 4096, 8192):
        qs = torch.randn(1, 2, SS, D, device=dev)
        ks = torch.randn(1, 2, SS, D, device=dev)
        vs = torch.randn(1, 2, SS, D, device=dev)
        d = measured_density(qs, ks, vs, sel_block=64, density=nom)
        print(f"  {SS:<10}{d:>34.4f}")
        assert abs(d - TARGET) < 0.02, f"density drifts with length: {d:.4f} at S={SS}"

    # 5. GRADIENT. The trainer backprops through this; top-k selection is non-differentiable, so
    #    confirm gradient still reaches q/k/v (it flows through the SELECTED values, as in MoBA).
    qg = q.clone().requires_grad_(True); kg = k.clone().requires_grad_(True)
    vg = v.clone().requires_grad_(True)
    nsa_attn(qg, kg, vg, window=64, n_sel=4, use_cmp=True).square().mean().backward()
    for nm, t in (("q", qg), ("k", kg), ("v", vg)):
        assert t.grad is not None and torch.isfinite(t.grad).all(), f"{nm} grad bad"
        print(f"  grad {nm}: |g|={t.grad.abs().mean().item():.3e}")
    # 6. DECODE. q_len=1 with kv_len=N through the transformers wrapper, compared against dense
    #    attention over the full KV.
    class _Mod:
        scaling = D ** -0.5
    fwd = make_nsa_attention(density=0.1438, sel_block=64)
    q1 = torch.randn(1, Hq, 1, D, device=dev)
    kN = torch.randn(1, Hkv, 977, D, device=dev)     # deliberately not a multiple of sel_block
    vN = torch.randn(1, Hkv, 977, D, device=dev)
    o_dec, _ = fwd(_Mod(), q1, kN, vN)
    kx = kN.repeat_interleave(Hq // Hkv, dim=1)
    vx = vN.repeat_interleave(Hq // Hkv, dim=1)
    o_ref = F.scaled_dot_product_attention(q1, kx, vx, scale=D ** -0.5,
                                           is_causal=False).transpose(1, 2)
    dd = (o_dec - o_ref).abs().max().item()
    print(f"  decode q_len=1 kv_len=977: shape {tuple(o_dec.shape)}, max diff vs dense {dd:.3e}")
    assert tuple(o_dec.shape) == (1, 1, Hq, D), f"decode returned {tuple(o_dec.shape)}"
    assert dd < 1e-5, f"decode path differs from dense over full KV ({dd:.3e})"

    print("  nsa_attn PASS (dense-reduction, causality, density, gradient, decode)")
