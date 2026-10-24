#!/usr/bin/env python3
"""
Wan2.2-TI2V-5B speed: dense vs INTRA on the same GPU and the same FlashAttention-2 kernel family.

  parity    fast path vs the reference path used for training/VBench ("evaluated"), both vs an
            fp32 masked oracle, per layer kind
  plumbing  the INTRA processor with dense attention reproduces stock diffusers (real weights)
  attn      per-layer attention latency (dense: every SDPA backend + raw FA2; INTRA: each layer
            kind including the permutation) and the 30-layer self-attention total
  dit       one transformer forward (real weights, random inputs, batch 1), dense vs INTRA
  e2e       one full video (text encoding + 50 steps x 2 CFG branches + VAE decoding)

Results are written to --out as JSON; a table is printed to stdout.
"""
import argparse, json, os, statistics, sys, time

import torch
import torch.nn.functional as F

import intra_wan as iw
from intra_wan_fast import fast_sparse_self_attention, layer_plan, fa2_dense, install_fast_intra

# name -> ((frames, height, width), token grid (T, H, W))
GRIDS = {
    "480p_49f": ((49, 480, 832), (13, 15, 26)),     # VBench setting
    "720p_121f": ((121, 704, 1280), (31, 22, 40)),  # native 5 s at 24 fps
    "720p_241f": ((241, 704, 1280), (61, 22, 40)),  # 10 s, standalone forward only
}
HEADS, HDIM, LAYERS = 24, 128, 30
DENSE_LAYERS = {0, LAYERS - 1}


def env_info():
    p = torch.cuda.get_device_properties(0)
    return dict(gpu=p.name, sm=f"sm{p.major}{p.minor}", torch=torch.__version__,
                cuda=torch.version.cuda, cudnn=torch.backends.cudnn.version())


def cuda_time(fn, iters, warmup):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    ts = []
    for _ in range(iters):
        a, b = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        a.record(); fn(); b.record()
        torch.cuda.synchronize()
        ts.append(a.elapsed_time(b))
    return dict(median_ms=statistics.median(ts), min_ms=min(ts), max_ms=max(ts), iters=iters)


def kernels_used(fn):
    """Top CUDA kernels that actually served a call (a requested SDPA backend is not proof)."""
    try:
        from torch.profiler import profile, ProfilerActivity
        with profile(activities=[ProfilerActivity.CUDA]) as prof:
            fn(); torch.cuda.synchronize()
        ev = sorted(prof.key_averages(), key=lambda e: -e.device_time_total)
        return [e.key[:90] for e in ev if e.device_time_total > 0][:3]
    except Exception as e:
        return [f"profiler failed: {type(e).__name__}"]


def qkv(N, heads, dev, dtype=torch.bfloat16, seed=0):
    """(B, H, N, D) views of contiguous (B, N, H, D) tensors, as the processor produces them."""
    g = torch.Generator(device=dev).manual_seed(seed)
    mk = lambda: torch.randn(1, N, heads, HDIM, device=dev, dtype=dtype, generator=g).transpose(1, 2)
    return mk(), mk(), mk()


def kind_label(li, cfg):
    kind = iw.layer_kind(li, cfg)
    suffix = ("-shift" if kind[1] else "0") if kind[0] == "gather" else ""
    return f"L{li % 5}:{kind[0]}{suffix}"


def oracle_mask(cfg, grid, li, dev):
    T, H, W = grid
    N = T * H * W
    groups = iw.layer_groups(cfg, grid, li, dev)
    if groups is None:
        R = iw.cone_radius(iw.layer_kind(li, cfg)[1], H * W)
        i = torch.arange(N, device=dev)
        return (i[:, None] - i[None, :]).abs() <= R
    gid = torch.empty(N, dtype=torch.long, device=dev)
    for g, idx in enumerate(groups):
        gid[idx] = g
    return gid[:, None] == gid[None, :]


# ------------------------------------------------------------------------------------------------
def cmd_parity(a):
    dev, cfg, res = "cuda", iw.INTRAConfig(), []
    for name in a.grids:
        _, grid = GRIDS[name]
        N = grid[0] * grid[1] * grid[2]
        heads = 2 if N > 20000 else 4          # the fp32 masked oracle is O(N^2) per head
        q, k, v = qkv(N, heads, dev)
        q32, k32, v32 = (x.float() for x in (q, k, v))
        for li in range(1, 6):                 # one period: gather, scatter-B, gather-shift, cone, scatter-A
            allow = oracle_mask(cfg, grid, li, dev)
            oracle = iw._dense_masked(q32, k32, v32, HDIM ** -0.5, allow)
            del allow
            fast = fast_sparse_self_attention(q, k, v, cfg, grid, li).float()
            ref = iw.sparse_self_attention(q, k, v, cfg, grid, li).float()
            e_fast = (fast - oracle).abs().max().item()
            e_ref = (ref - oracle).abs().max().item()
            e_fr = (fast - ref).abs().max().item()
            ok = e_fast <= max(2 * e_ref, 2e-2)
            res.append(dict(grid=name, layer=li, kind=str(iw.layer_kind(li, cfg)), err_fast_vs_fp32=e_fast,
                            err_evaluated_vs_fp32=e_ref, err_fast_vs_evaluated=e_fr, ok=ok))
            print(f"  {name:10s} {kind_label(li, cfg):16s} |fast-fp32|={e_fast:.2e} "
                  f"|ref-fp32|={e_ref:.2e} |fast-ref|={e_fr:.2e}  {'OK' if ok else 'FAIL'}", flush=True)
            del oracle, fast, ref
            torch.cuda.empty_cache()
        qd, kd, vd = qkv(N, HEADS if N <= 20000 else 4, dev, seed=1)
        sdpa = F.scaled_dot_product_attention(qd, kd, vd)
        got = fa2_dense(qd.transpose(1, 2), kd.transpose(1, 2), vd.transpose(1, 2), HDIM ** -0.5).transpose(1, 2)
        e = (sdpa - got).abs().max().item()
        res.append(dict(grid=name, layer="dense", err_fa2_vs_sdpa=e, ok=e < 1e-2))
        print(f"  {name:10s} dense            |fa2-sdpa|={e:.2e}", flush=True)
    allok = all(r["ok"] for r in res)
    print("PARITY", "PASS" if allok else "FAIL", flush=True)
    json.dump(dict(env=env_info(), results=res, ok=allok), open(os.path.join(a.out, "parity.json"), "w"), indent=1)
    if not allok:
        sys.exit(1)


# ------------------------------------------------------------------------------------------------
def load_transformer(model, dev):
    from diffusers import WanTransformer3DModel
    return WanTransformer3DModel.from_pretrained(model, subfolder="transformer",
                                                 torch_dtype=torch.bfloat16).to(dev).eval()


def set_arm(tf, arm, grid):
    from diffusers.models.transformers.transformer_wan import WanAttnProcessor
    for b in tf.blocks:
        b.attn1.set_processor(WanAttnProcessor())
    if arm == "intra":
        install_fast_intra(tf, iw.INTRAConfig(), grid, DENSE_LAYERS)


def cmd_plumbing(a):
    """Our processor with dense attention in place of INTRA must match stock diffusers: checks the
    q/k/v, QK-norm, RoPE and output-projection code against the installed diffusers version."""
    dev = "cuda"
    tf = load_transformer(a.model, dev)
    C = tf.config.in_channels
    torch.manual_seed(0)
    x = torch.randn(1, C, 4, 16, 16, device=dev, dtype=torch.bfloat16)      # grid (4, 8, 8)
    ts = torch.tensor([500.0], device=dev)
    ctx = torch.randn(1, 512, tf.config.text_dim, device=dev, dtype=torch.bfloat16) * 0.1
    run = lambda: tf(x, ts, ctx, return_dict=False)[0].float()
    with torch.no_grad():
        set_arm(tf, "dense", None)
        ref = run()
        dense_fn = lambda q, k, v, cfg, grid, li: F.scaled_dot_product_attention(q, k, v)
        iw.install_intra_processors(tf, iw.INTRAConfig(), dense_layers=(), attn_fn=dense_fn)
        iw.set_intra_grid(tf, (4, 8, 8))
        got = run()
    err = (ref - got).abs().max().item()
    ok = err < 5e-3
    print(f"max|stock - INTRA processor (dense core)| = {err:.3e}  {'PASS' if ok else 'FAIL'}", flush=True)
    sys.exit(0 if ok else 1)


# ------------------------------------------------------------------------------------------------
def cmd_attn(a):
    from torch.nn.attention import sdpa_kernel, SDPBackend
    dev, cfg = "cuda", iw.INTRAConfig()
    out = dict(env=env_info(), grids={})
    for name in a.grids:
        (frames, hh, ww), grid = GRIDS[name]
        N = grid[0] * grid[1] * grid[2]
        q, k, v = qkv(N, HEADS, dev)
        scale = HDIM ** -0.5
        row = dict(frames=frames, height=hh, width=ww, grid=grid, N=N, dense={}, intra={})
        # dense: every SDPA backend that runs, plus raw FA2; the baseline is the fastest
        for bname, be in [("sdpa_flash", SDPBackend.FLASH_ATTENTION), ("sdpa_cudnn", SDPBackend.CUDNN_ATTENTION),
                          ("sdpa_efficient", SDPBackend.EFFICIENT_ATTENTION)]:
            def f(be=be):
                with sdpa_kernel([be]):
                    return F.scaled_dot_product_attention(q, k, v)
            try:
                t = cuda_time(f, a.iters, a.warmup)
                t["kernels"] = kernels_used(f)
                row["dense"][bname] = t
            except Exception as e:
                row["dense"][bname] = dict(error=f"{type(e).__name__}: {str(e)[:100]}")
        qn, kn, vn = (x.transpose(1, 2) for x in (q, k, v))
        f = lambda: fa2_dense(qn, kn, vn, scale)
        row["dense"]["fa2_raw"] = cuda_time(f, a.iters, a.warmup)
        row["dense"]["fa2_raw"]["kernels"] = kernels_used(f)
        ok = {b: t["median_ms"] for b, t in row["dense"].items() if "median_ms" in t}
        best = min(ok, key=ok.get)
        row["dense_best"] = best
        dense_ms = ok[best]
        # INTRA: each layer kind through the full fast path (permute + FA2 + inverse permute)
        per_kind = {}
        for li in range(1, 6):
            plan = layer_plan(cfg, grid, li, 1, dev)
            f = lambda li=li: fast_sparse_self_attention(q, k, v, cfg, grid, li)
            t = cuda_time(f, a.iters, a.warmup)
            t["kernels"] = kernels_used(f)
            t["density"] = plan[-1] / N / N
            if plan[0] == "group":        # attention alone, to show what the permutation costs
                _, perm, inv, cu, mx, _ = plan
                fl = lambda x: x.reshape(N, HEADS, HDIM).index_select(0, perm)
                pq, pk, pv = fl(qn), fl(kn), fl(vn)
                t["attn_only_ms"] = cuda_time(
                    lambda: torch.ops.aten._flash_attention_forward.default(
                        pq, pk, pv, cu, cu, mx, mx, 0.0, False, False, scale=scale)[0],
                    a.iters, a.warmup)["median_ms"]
                t["n_groups"] = int(cu.numel() - 1)
                t["max_group"] = int(mx)
            else:
                t["window_R"] = plan[1]
            row["intra"][kind_label(li, cfg)] = t
            per_kind[li % 5] = t
        # 30-layer total: blocks 0 and 29 dense, the rest follow the period-5 schedule
        stack_ms, stack_pairs = 0.0, 0
        for li in range(LAYERS):
            if li in DENSE_LAYERS:
                stack_ms += dense_ms; stack_pairs += N * N
            else:
                stack_ms += per_kind[li % 5]["median_ms"]; stack_pairs += per_kind[li % 5]["density"] * N * N
        row["stack"] = dict(dense_ms=LAYERS * dense_ms, intra_ms=stack_ms,
                            speedup=LAYERS * dense_ms / stack_ms,
                            density=stack_pairs / (LAYERS * N * N),
                            ideal_speedup=(LAYERS * N * N) / stack_pairs)
        out["grids"][name] = row
        print(f"\n=== {name}  grid={grid}  N={N}  (1 attention call, B=1, {HEADS} heads, bf16) ===", flush=True)
        for b, t in row["dense"].items():
            s = f"{t['median_ms']:8.2f} ms" if "median_ms" in t else t["error"]
            print(f"  dense {b:15s} {s}   {(t.get('kernels') or [''])[0]}", flush=True)
        print(f"  -> dense baseline = {best} ({dense_ms:.2f} ms)", flush=True)
        for kl, t in row["intra"].items():
            extra = (f"attn-only {t['attn_only_ms']:.2f} ms, {t['n_groups']} groups" if "attn_only_ms" in t
                     else f"window R={t['window_R']}")
            print(f"  intra {kl:15s} {t['median_ms']:8.2f} ms  density {t['density']:.3f}  "
                  f"{dense_ms / t['median_ms']:5.2f}x   ({extra})", flush=True)
        st = row["stack"]
        print(f"  30-layer stack: dense {st['dense_ms']:.1f} ms  intra {st['intra_ms']:.1f} ms  "
              f"speedup {st['speedup']:.2f}x  (density {st['density']:.3f}, ideal {st['ideal_speedup']:.2f}x)", flush=True)
        del q, k, v
        torch.cuda.empty_cache()
    json.dump(out, open(os.path.join(a.out, "attn.json"), "w"), indent=1)


# ------------------------------------------------------------------------------------------------
def cmd_dit(a):
    dev = "cuda"
    tf = load_transformer(a.model, dev)
    out = dict(env=env_info(), grids={})
    for name in a.grids:
        _, grid = GRIDS[name]
        T, H, W = grid
        N = T * H * W
        g = torch.Generator(device=dev).manual_seed(0)
        x = torch.randn(1, 48, T, 2 * H, 2 * W, device=dev, dtype=torch.bfloat16, generator=g)
        ctx = torch.randn(1, 512, 4096, device=dev, dtype=torch.bfloat16, generator=g)
        ts = torch.full((1, N), 500.0, device=dev)          # Wan2.2 TI2V uses per-token timesteps
        fwd = lambda: tf(hidden_states=x, timestep=ts, encoder_hidden_states=ctx, return_dict=False)[0]
        row = {}
        with torch.no_grad():
            for arm in a.arms:                  # dense, intra, dense, intra: also shows drift
                set_arm(tf, arm, grid)
                t = cuda_time(fwd, a.iters, a.warmup)
                row.setdefault(arm, []).append(t)
                print(f"  {name:10s} {arm:6s} forward {t['median_ms']:9.1f} ms  (min {t['min_ms']:.1f}, max {t['max_ms']:.1f})", flush=True)
        d = min(r["median_ms"] for r in row["dense"])
        i = min(r["median_ms"] for r in row["intra"])
        row["speedup"] = d / i
        row["N"] = N
        out["grids"][name] = row
        print(f"  {name:10s} DiT forward speedup {d / i:.2f}x  (dense {d:.1f} ms -> intra {i:.1f} ms)", flush=True)
        del x, ctx
        torch.cuda.empty_cache()
    json.dump(out, open(os.path.join(a.out, "dit.json"), "w"), indent=1)


# ------------------------------------------------------------------------------------------------
def cmd_e2e(a):
    from diffusers import WanPipeline
    from diffusers.utils import export_to_video
    from PIL import Image
    dev = "cuda"
    (frames, hh, ww), grid = GRIDS[a.grids[0]]
    pipe = WanPipeline.from_pretrained(a.model, torch_dtype=torch.bfloat16).to(dev)
    pipe.set_progress_bar_config(disable=True)
    set_arm(pipe.transformer, a.arm, grid)
    zd = pipe.vae.config.z_dim
    vmean = torch.tensor(pipe.vae.config.latents_mean, device=dev).view(1, zd, 1, 1, 1)
    vstd = torch.tensor(pipe.vae.config.latents_std, device=dev).view(1, zd, 1, 1, 1)

    # wall time spent inside transformer calls, to separate the DiT from text encoding / glue
    tf_ms = []
    orig_fwd = pipe.transformer.forward
    def timed_fwd(*args, **kw):
        torch.cuda.synchronize(); t0 = time.perf_counter()
        r = orig_fwd(*args, **kw)
        torch.cuda.synchronize(); tf_ms.append((time.perf_counter() - t0) * 1e3)
        return r
    pipe.transformer.forward = timed_fwd

    runs = []
    for rep in range(a.reps + 1):              # rep 0 is warm-up
        tf_ms.clear()
        g = torch.Generator(device=dev).manual_seed(20260926)
        torch.cuda.synchronize(); t0 = time.perf_counter()
        with torch.no_grad():
            o = pipe(prompt=a.prompt, height=hh, width=ww, num_frames=frames,
                     num_inference_steps=a.steps, guidance_scale=a.guidance,
                     output_type="latent", generator=g)
            lat = (o.frames if hasattr(o, "frames") else o[0]).float()
            torch.cuda.synchronize(); t1 = time.perf_counter()
            z = (lat.to(dev, torch.float32) * vstd + vmean).to(pipe.vae.dtype)
            vid = pipe.vae.decode(z, return_dict=False)[0]
            torch.cuda.synchronize(); t2 = time.perf_counter()
        r = dict(rep=rep, total_s=t2 - t0, pipe_s=t1 - t0, vae_decode_s=t2 - t1,
                 transformer_s=sum(tf_ms) / 1e3, transformer_calls=len(tf_ms),
                 per_forward_ms=statistics.median(tf_ms))
        r["other_s"] = r["pipe_s"] - r["transformer_s"]      # text encoding + scheduler + CFG glue
        runs.append(r)
        print(f"  [{a.arm} {a.grids[0]}] rep {rep}: total {r['total_s']:.1f}s = DiT {r['transformer_s']:.1f}s "
              f"({r['transformer_calls']} fwd x {r['per_forward_ms']:.0f} ms) + VAE {r['vae_decode_s']:.1f}s "
              f"+ other {r['other_s']:.1f}s", flush=True)
    v = ((vid[0].permute(1, 0, 2, 3).float() + 1.0) / 2.0).clamp(0, 1)
    fr = [Image.fromarray((f.permute(1, 2, 0).cpu().numpy() * 255).astype("uint8")) for f in v]
    export_to_video(fr, os.path.join(a.out, f"e2e_{a.arm}_{a.grids[0]}.mp4"), fps=24 if frames > 49 else 16)
    json.dump(dict(env=env_info(), arm=a.arm, grid=a.grids[0], prompt=a.prompt, steps=a.steps,
                   guidance=a.guidance, runs=runs),
              open(os.path.join(a.out, f"e2e_{a.arm}_{a.grids[0]}.json"), "w"), indent=1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["parity", "plumbing", "attn", "dit", "e2e"])
    ap.add_argument("--out", required=True)
    ap.add_argument("--model", default="Wan-AI/Wan2.2-TI2V-5B-Diffusers", help="HF id or local dir")
    ap.add_argument("--grids", default="480p_49f,720p_121f,720p_241f")
    ap.add_argument("--iters", type=int, default=20)
    ap.add_argument("--warmup", type=int, default=5)
    ap.add_argument("--arms", default="dense,intra,dense,intra", help="dit mode")
    ap.add_argument("--arm", default="dense", choices=["dense", "intra"], help="e2e mode")
    ap.add_argument("--reps", type=int, default=1, help="e2e timed repetitions after one warm-up")
    ap.add_argument("--steps", type=int, default=50)
    ap.add_argument("--guidance", type=float, default=5.0)
    ap.add_argument("--prompt", default="A red fox trotting through fresh snow in a birch forest at "
                                        "sunrise, cinematic tracking shot, highly detailed, realistic motion")
    a = ap.parse_args()
    a.grids = a.grids.split(",")
    a.arms = a.arms.split(",")
    os.makedirs(a.out, exist_ok=True)
    torch.backends.cuda.matmul.allow_tf32 = True
    dict(parity=cmd_parity, plumbing=cmd_plumbing, attn=cmd_attn, dit=cmd_dit, e2e=cmd_e2e)[a.mode](a)


if __name__ == "__main__":
    main()
