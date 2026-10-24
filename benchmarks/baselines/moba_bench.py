#!/usr/bin/env python3
"""Official MoBA prefill attention latency against the same dense baseline as bench_kernel.py.

Per attention layer, causal prefill, bf16, batch 1, LLaMA-3.1-8B shapes (32 q / 8 kv heads, d=128).
MoBA has no GQA path, so it (and dense SDPA) run on keys/values expanded to the query heads.
The chunk size is swept and MoBA is credited with its fastest configuration whose density does
not exceed the target (0.125, matching INTRA's g2s4 stack); top-k is kept >= 2 because MoBA falls
back to plain dense attention when its effective top-k reaches 0.

    git clone https://github.com/MoonshotAI/MoBA && MOBA_SRC=$PWD/MoBA python moba_bench.py --out moba.json
"""
import argparse, json, os, sys, time
import torch


def timeit(fn, warmup, iters):
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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seqlens", default="4096,16384,65536")
    ap.add_argument("--heads", type=int, default=32)
    ap.add_argument("--kv-heads", type=int, default=8)
    ap.add_argument("--dim", type=int, default=128)
    ap.add_argument("--moba-density", type=float, default=0.125)
    ap.add_argument("--iters", type=int, default=50)
    ap.add_argument("--warmup", type=int, default=10)
    ap.add_argument("--out", default="results_moba_speed.json")
    a = ap.parse_args()

    dev, dt = "cuda", torch.bfloat16
    torch.backends.cuda.matmul.allow_tf32 = True

    # moba_efficient.py is loaded by file path: the package __init__ only registers MoBA with
    # transformers' attention dispatch, which this benchmark does not need.
    import importlib.util
    _mp = os.path.join(os.environ["MOBA_SRC"], "moba", "moba_efficient.py")
    if not os.path.exists(_mp):
        raise SystemExit(f"!! official MoBA not found at {_mp}")
    _spec = importlib.util.spec_from_file_location("moba_efficient", _mp)
    _mod = importlib.util.module_from_spec(_spec)
    _spec.loader.exec_module(_mod)

    # MoBA was written against flash-attn 2.6.3, whose _flash_attn_varlen_forward returned an
    # 8-tuple (out, q, k, v, out_padded, softmax_lse, S_dmask, rng_state); current flash-attn
    # returns (out, softmax_lse, S_dmask, rng_state). Re-shape the tuple instead of pinning the
    # old kernel; out == out_padded whenever head_dim is a multiple of 8.
    assert a.dim % 8 == 0, f"head dim {a.dim} is not a multiple of 8; out vs out_padded would differ"
    _orig_fwd = _mod._flash_attn_varlen_forward

    def _fwd_compat(*args, **kwargs):
        r = _orig_fwd(*args, **kwargs)
        if isinstance(r, tuple) and len(r) == 8:
            return r
        if isinstance(r, tuple) and len(r) >= 2:
            out, lse = r[0], r[1]
            rest = list(r[2:]) + [None, None]
            return (None, None, None, None, out, lse, rest[0], rest[1])
        raise RuntimeError(f"unexpected _flash_attn_varlen_forward return: {type(r)} len "
                           f"{len(r) if isinstance(r, tuple) else 'n/a'}")

    _mod._flash_attn_varlen_forward = _fwd_compat
    moba_attn_varlen = _mod.moba_attn_varlen
    print("  official MoBA imported from:", _mp)
    import flash_attn
    print("  flash_attn:", getattr(flash_attn, "__version__", "unknown"))

    out = {"gpu": torch.cuda.get_device_name(0), "torch": torch.__version__, "moba_density": a.moba_density, "rows": []}

    for S in [int(x) for x in a.seqlens.split(",")]:
        H, Hk, Dh = a.heads, a.kv_heads, a.dim
        q = torch.randn(1, H, S, Dh, device=dev, dtype=dt)
        k = torch.randn(1, Hk, S, Dh, device=dev, dtype=dt)
        v = torch.randn(1, Hk, S, Dh, device=dev, dtype=dt)
        row = {"S": S}

        # ---- dense: the single shared denominator
        kx = k.repeat_interleave(H // Hk, dim=1)
        vx = v.repeat_interleave(H // Hk, dim=1)
        row["dense"] = timeit(
            lambda: torch.nn.functional.scaled_dot_product_attention(q, kx, vx, is_causal=True),
            a.warmup, a.iters)

        # ---- official MoBA, varlen layout [tokens, heads, dim]. Sweep chunk sizes, keep configs whose
        # density does not exceed the target, report the fastest (or the cheapest over-target one,
        # flagged by `over_target`, if none can reach it without top-k degenerating).
        qv = q[0].transpose(0, 1).contiguous()
        kv_ = kx[0].transpose(0, 1).contiguous()   # expanded to H heads: MoBA has no GQA path
        vv = vx[0].transpose(0, 1).contiguous()
        cu = torch.tensor([0, S], device=dev, dtype=torch.int32)

        cands, seen_cfg = [], set()
        for chunk in (4096, 2048, 1024, 512, 256, 128):
            if chunk >= S:
                continue
            nb = S // chunk
            if nb < 2:
                continue
            topk = max(2, round(a.moba_density * nb))     # 2 is the smallest non-degenerate value
            if (chunk, topk) in seen_cfg:
                continue
            seen_cfg.add((chunk, topk))
            dens = topk / nb
            cands.append({"chunk": chunk, "blocks": nb, "topk": topk, "density": dens,
                          "over_target": dens > a.moba_density * 1.02})
        results = []
        for c in cands:
            try:
                t = timeit(lambda c=c: moba_attn_varlen(qv, kv_, vv, cu, S,
                                                       moba_chunk_size=c["chunk"], moba_topk=c["topk"]),
                           a.warmup, a.iters)
                results.append({**c, **t})
            except Exception as e:
                results.append({**c, "error": f"{type(e).__name__}: {e}"[:160]})
        row["moba_sweep"] = results
        ok = [r for r in results if "median_ms" in r]
        at_target = [r for r in ok if not r["over_target"]]
        best = min(at_target or ok, key=lambda r: r["median_ms"]) if ok else None
        if best is None:
            row["moba_error"] = "; ".join(r.get("error", "") for r in results)[:200] or "no candidates"
            print(f"  !! S={S} MoBA failed: {row['moba_error']}")
        else:
            row["moba"] = {k: best[k] for k in ("median_ms", "min_ms", "p90_ms", "iters")}
            row["moba_topk"] = best["topk"]; row["moba_blocks"] = best["blocks"]
            row["moba_chunk"] = best["chunk"]; row["moba_eff_density"] = best["density"]
            row["moba_over_target"] = best["over_target"]
            for r in sorted(ok, key=lambda r: r["median_ms"]):
                mark = " <- best at target" if r is best else (" (over target)" if r["over_target"] else "")
                print(f"      S={S:>6} chunk {r['chunk']:>5} topk {r['topk']:>3}/{r['blocks']:<4}"
                      f" dens {r['density']:.3f}  {r['median_ms']:8.3f} ms{mark}")

        out["rows"].append(row)
        d = row["dense"]["median_ms"]
        m = row.get("moba", {}).get("median_ms")
        print(f"  S={S:>6}  dense {d:8.3f} ms   MoBA {m if m is None else round(m,3)} ms"
              f"   -> {'n/a' if m is None else f'{d/m:.2f}x'}"
              f"   (topk {row.get('moba_topk')}/{row.get('moba_blocks')}"
              f" = {row.get('moba_eff_density', float('nan')):.3f}, chunk {row.get('moba_chunk')})")

    with open(a.out, "w") as f:
        json.dump(out, f, indent=1)
    print("\nwrote", a.out)


if __name__ == "__main__":
    main()
