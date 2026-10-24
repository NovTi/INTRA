#!/usr/bin/env python3
"""NSA prefill attention latency with the same shapes and dense baseline as moba_bench.py.

    git clone https://github.com/Relaxed-System-Lab/Flash-Sparse-Attention && cd Flash-Sparse-Attention
    git checkout 1325e8d && git apply ../fsa_triton36.patch     # one-line fix for triton >= 3.6
    FSA_SRC=$PWD python ../nsa_bench.py --out nsa.json

Kernels: Flash-Sparse-Attention (Relaxed-System-Lab, commit 1325e8d) -- its nsa_ref/ops are the
XunhaoLai native-sparse-attention-triton kernels (compressed + top-k selected attention), and its
fsa/ops add the FSA selected-attention kernel built for GQA groups < 8 (LLaMA-3.1-8B is 32q/8kv = 4).
The fla-org NSA kernel is not usable here: it asserts GQA group % 16 == 0.

Timed = everything NSA computes from q,k,v, as in FSA's own module forward:
  compress k,v (avg-pool; the cheapest compressor and the one our quality arm used)
  + compressed attention (incl. block scoring and top-k selection)
  + selected-block attention (the faster of the NSA and FSA kernels, credited to NSA)
  + sliding-window attention (upstream flash-attn varlen, native GQA)
  + gate-weighted sum of the three branch outputs.
Not timed, for any method: q/k/v/o projections, RoPE, the gate projection.

Density follows experiments/llm (NSA student): attended keys per query (window U selected blocks, plus
compressed keys) over the causal context, summed over queries -- the same denominator INTRA's 1/8
Scatter and MoBA's topk/blocks use. It is measured from the top-k indices the kernel actually picked.

Two NSA rows per length:
  published   l=32, d=16, l'=64, n=16, w=512 (the paper's hyper-parameters, density = whatever results)
  iso-density every kernel-legal (l, d, l', w) split, n solved to the largest value whose MEASURED
              density does not exceed the target; NSA is credited with the FASTEST of them -- the
              same rule moba_bench.py uses for MoBA's chunk size.
"""
import argparse, json, os, sys, time

import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.environ.get("FSA_SRC", os.path.join(HERE, "Flash-Sparse-Attention")))
from flash_attn import flash_attn_varlen_func  # noqa: E402
from nsa_ref.ops import avgpool_compress, compressed_attention, topk_sparse_attention  # noqa: E402
from fsa.ops import FSA_topk_sparse_attention  # noqa: E402


def timeit(fn, warmup, iters):          # identical to moba_bench.py
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    ts = []
    for _ in range(iters):
        torch.cuda.synchronize(); t0 = time.perf_counter()
        fn()
        torch.cuda.synchronize(); ts.append((time.perf_counter() - t0) * 1e3)
    ts.sort()
    return {"median_ms": ts[len(ts) // 2], "min_ms": ts[0], "p90_ms": ts[int(len(ts) * 0.9)], "iters": iters}


SEL = {"nsa": topk_sparse_attention, "fsa": FSA_topk_sparse_attention}


def nsa_forward(q, k, v, gate, cu, S, c, sel="fsa", return_idx=False):
    ck, ccu = avgpool_compress(k, None, cu, c["l"], c["d"])
    cv, _ = avgpool_compress(v, None, cu, c["l"], c["d"])
    # parallel top-k is the faster path (~1 ms at 16K); the library itself disables it above 32K
    # ("avoid some current bugs when sequence length is too long"), so NSA gets it wherever it is allowed.
    o_c, idx = compressed_attention(q, ck, cv, c["l"], c["d"], c["B"], c["n"], cu, ccu, S,
                                    int(ck.shape[0]), None, 1, 2, parallel_topk_compute=S <= 32768)
    o_s = SEL[sel](q, k, v, idx, c["B"], cu, None)
    o_w = flash_attn_varlen_func(q, k, v, cu, cu, S, S, causal=True, window_size=(c["w"], -1))
    o = gate[..., 0:1] * o_c + gate[..., 1:2] * o_s + gate[..., 2:3] * o_w
    return (o, idx) if return_idx else o


def measured_density(idx, S, c):
    """Exact causal density of the pattern NSA ran, averaged over kv heads."""
    dev = idx.device
    i = torch.arange(S, device=dev, dtype=torch.int64)
    ctx = (i + 1).double()
    win_lo = torch.clamp(i - c["w"], min=0)                                # FA window (w, -1), causal
    win = (i - win_lo + 1).double()
    st = idx.long() * c["B"]                                               # (Hk, S, n)
    valid = (idx >= 0) & (st <= i[None, :, None])
    en = torch.minimum(st + c["B"], (i + 1)[None, :, None])
    sel = torch.where(valid, en - st, 0)
    ov = torch.where(valid, torch.clamp(torch.minimum(en, (i + 1)[None, :, None])
                                        - torch.maximum(st, win_lo[None, :, None]), min=0), 0)
    full = win[None] + (sel - ov).sum(-1).double()                          # window U selected
    ncmp = torch.clamp(torch.div(i + 1 - c["l"], c["d"], rounding_mode="floor") + 1, min=0).double()
    return float(((full + ncmp[None]).sum(-1) / ctx.sum()).mean())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seqlens", default="4096,16384,65536")
    ap.add_argument("--heads", type=int, default=32)
    ap.add_argument("--kv-heads", type=int, default=8)
    ap.add_argument("--dim", type=int, default=128)
    ap.add_argument("--target", type=float, default=0.125)
    ap.add_argument("--iters", type=int, default=50)
    ap.add_argument("--warmup", type=int, default=10)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    dev, dt = "cuda", torch.bfloat16
    torch.backends.cuda.matmul.allow_tf32 = True
    out = {"gpu": torch.cuda.get_device_name(0), "torch": torch.__version__,
           "target_density": a.target, "rows": []}

    for S in [int(x) for x in a.seqlens.split(",")]:
        H, Hk, Dh = a.heads, a.kv_heads, a.dim
        torch.manual_seed(20260926)
        q = torch.randn(1, H, S, Dh, device=dev, dtype=dt)
        k = torch.randn(1, Hk, S, Dh, device=dev, dtype=dt)
        v = torch.randn(1, Hk, S, Dh, device=dev, dtype=dt)
        row = {"S": S}
        # ---- dense: same denominator as moba_bench.py (SDPA causal, kv expanded to H heads)
        kx, vx = k.repeat_interleave(H // Hk, 1), v.repeat_interleave(H // Hk, 1)
        row["dense"] = timeit(lambda: torch.nn.functional.scaled_dot_product_attention(q, kx, vx, is_causal=True),
                              a.warmup, a.iters)
        del kx, vx
        qv, kv_, vv = (x[0].transpose(0, 1).contiguous() for x in (q, k, v))   # varlen [S, heads, D]
        cu = torch.tensor([0, S], device=dev, dtype=torch.int32)
        gate = torch.sigmoid(torch.randn(S, H, 3, device=dev, dtype=dt))
        # upstream FA2 dense with native GQA, for reference (NSA's window branch uses this kernel)
        row["dense_fa2_gqa"] = timeit(lambda: flash_attn_varlen_func(qv, kv_, vv, cu, cu, S, S, causal=True),
                                      a.warmup, a.iters)

        def run(c):
            """Density + both selected-attention kernels; NSA is credited with the faster one."""
            o_f, idx = nsa_forward(qv, kv_, vv, gate, cu, S, c, "fsa", return_idx=True)
            o_n = nsa_forward(qv, kv_, vv, gate, cu, S, c, "nsa")
            assert torch.isfinite(o_f).all() and torch.isfinite(o_n).all(), "non-finite NSA output"
            agree = (o_f.float() - o_n.float()).abs().max().item()
            r = dict(c, density=measured_density(idx, S, c), fsa_vs_nsa_maxdiff=agree)
            for s in ("fsa", "nsa"):
                r[s] = timeit(lambda s=s: nsa_forward(qv, kv_, vv, gate, cu, S, c, s), a.warmup, a.iters)
            r["best_kernel"] = min(("fsa", "nsa"), key=lambda s: r[s]["median_ms"])
            r["median_ms"] = r[r["best_kernel"]]["median_ms"]
            return r

        def density_of(c):
            _, idx = nsa_forward(qv, kv_, vv, gate, cu, S, c, "fsa", return_idx=True)
            return measured_density(idx, S, c)

        # ---- published hyper-parameters
        pub = run(dict(l=32, d=16, B=64, n=16, w=512))
        row["nsa_published"] = pub
        print(f"  S={S:>6} NSA published l32 d16 B64 n16 w512  dens {pub['density']:.3f}  "
              f"{pub['median_ms']:8.3f} ms ({pub['best_kernel']}; fsa {pub['fsa']['median_ms']:.3f} / "
              f"nsa {pub['nsa']['median_ms']:.3f}; |fsa-nsa|={pub['fsa_vs_nsa_maxdiff']:.1e})", flush=True)

        # ---- iso-density sweep
        sweep = []
        for (l, d) in ((32, 16), (64, 32), (128, 64)):
            for B in (64, 128):
                for w in (512, 256, 128):
                    base = dict(l=l, d=d, B=B, w=w)
                    lo, hi = 3, S // B                       # n >= init(1) + local(2)
                    if density_of(dict(base, n=lo)) > a.target * 1.02:
                        continue                             # this split cannot reach the target
                    while lo < hi:                           # largest n with density <= target
                        mid = (lo + hi + 1) // 2
                        if density_of(dict(base, n=mid)) <= a.target * 1.02:
                            lo = mid
                        else:
                            hi = mid - 1
                    try:
                        r = run(dict(base, n=lo))
                        sweep.append(r)
                        print(f"      S={S:>6} l{l:<3} d{d:<3} B{B:<4} w{w:<4} n{lo:<4} dens {r['density']:.3f}  "
                              f"{r['median_ms']:8.3f} ms ({r['best_kernel']})", flush=True)
                    except Exception as e:
                        sweep.append(dict(base, n=lo, error=f"{type(e).__name__}: {e}"[:160]))
                        print(f"      S={S:>6} l{l} d{d} B{B} w{w} n{lo}  ERROR {sweep[-1]['error']}", flush=True)
        row["nsa_iso_sweep"] = sweep
        ok = [r for r in sweep if "median_ms" in r]
        if ok:
            best = min(ok, key=lambda r: r["median_ms"])
            row["nsa_iso"] = best
            print(f"  S={S:>6} NSA iso-density best: l{best['l']} d{best['d']} B{best['B']} w{best['w']} "
                  f"n{best['n']} dens {best['density']:.3f}  {best['median_ms']:.3f} ms", flush=True)
        dn = row["dense"]["median_ms"]
        print(f"  S={S:>6} dense {dn:.3f} ms | NSA published {pub['median_ms']:.3f} ms ({dn/pub['median_ms']:.2f}x)"
              + (f" | NSA iso {row['nsa_iso']['median_ms']:.3f} ms ({dn/row['nsa_iso']['median_ms']:.2f}x)" if ok else ""),
              flush=True)
        out["rows"].append(row)
        del q, k, v, qv, kv_, vv, gate
        torch.cuda.empty_cache()
    json.dump(out, open(a.out, "w"), indent=1)
    print("wrote", a.out)


if __name__ == "__main__":
    main()
