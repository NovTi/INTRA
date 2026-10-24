"""Attention-only latency of the INTRA kernels against dense FlashAttention-2.

Dense is PyTorch SDPA pinned to its FlashAttention-2 backend, with GQA keys/values expanded to the
query heads. Latency is the median over --iters timed calls (CUDA events) after --warmup calls.

    python benchmarks/bench_kernel.py llm   --seqlens 4096,8192,16384,32768,65536
    python benchmarks/bench_kernel.py image     # 1024^2 and 2048^2 FLUX images

`llm` also prints the 32-layer total of the retrieval schedule used in the paper (g2s4: Scatter
G=2 in 4 layers, width-576 windows in the other 28), i.e. the INTRA column of Table 17(a).
"""
import argparse
import json

import torch
import torch.nn.functional as F
from torch.nn.attention import SDPBackend, sdpa_kernel

from intra_attn import intra_image_attn, intra_llm_attn
from intra_attn.reference import TEXT_LEN


def median_ms(fn, warmup, iters):
    for _ in range(warmup):
        fn()
    times = []
    for _ in range(iters):
        start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        start.record()
        fn()
        end.record()
        torch.cuda.synchronize()
        times.append(start.elapsed_time(end))
    return sorted(times)[len(times) // 2]


def dense_fa2(q, k, v, causal):
    rep = q.shape[2] // k.shape[2]
    q, k, v = (x.transpose(1, 2) for x in (q, k, v))
    k, v = k.repeat_interleave(rep, 1), v.repeat_interleave(rep, 1)
    with sdpa_kernel(SDPBackend.FLASH_ATTENTION):
        return F.scaled_dot_product_attention(q, k, v, is_causal=causal)


def qkv(B, S, H, Hkv):
    q = torch.randn(B, S, H, 128, device="cuda", dtype=torch.bfloat16)
    k = torch.randn(B, S, Hkv, 128, device="cuda", dtype=torch.bfloat16)
    v = torch.randn(B, S, Hkv, 128, device="cuda", dtype=torch.bfloat16)
    return q, k, v


@torch.no_grad()
def bench_llm(args):
    rows = []
    for S in args.seqlens:
        q, k, v = qkv(args.batch, S, args.heads, args.kv_heads)
        r = {"seqlen": S, "dense": median_ms(lambda: dense_fa2(q, k, v, True), args.warmup, args.iters)}
        for G in args.groups:
            r[f"scatter_g{G}"] = median_ms(lambda: intra_llm_attn(q, k, v, "scatter", group_num=G),
                                           args.warmup, args.iters)
        if S % 64 == 0:
            r["gather1_g8"] = median_ms(lambda: intra_llm_attn(q, k, v, "gather1", group_num=8),
                                        args.warmup, args.iters)
        r["window576"] = median_ms(lambda: intra_llm_attn(q, k, v, "window", window=576),
                                   args.warmup, args.iters)
        if "scatter_g2" in r:
            r["g2s4_32layer"] = 4 * r["scatter_g2"] + 28 * r["window576"]
            r["dense_32layer"] = 32 * r["dense"]
        rows.append(r)
        print("  ".join(f"{k_}={v_:.3f}" if isinstance(v_, float) else f"{k_}={v_}" for k_, v_ in r.items()),
              flush=True)
    return rows


@torch.no_grad()
def bench_image(args):
    rows = []
    for width in (64, 128):  # 1024^2 and 2048^2 FLUX images
        S = TEXT_LEN + width * width
        q, k, v = qkv(args.batch, S, args.image_heads, args.image_heads)
        r = {"width": width, "seqlen": S,
             "dense": median_ms(lambda: dense_fa2(q, k, v, False), args.warmup, args.iters)}
        for pattern in ("scatter", "gather0"):
            r[pattern] = median_ms(lambda: intra_image_attn(q, k, v, pattern, width), args.warmup, args.iters)
        r["interleaved_mean"] = (r["scatter"] + r["gather0"]) / 2  # layers alternate the two patterns
        rows.append(r)
        print("  ".join(f"{k_}={v_:.3f}" if isinstance(v_, float) else f"{k_}={v_}" for k_, v_ in r.items()))
    return rows


def main():
    p = argparse.ArgumentParser()
    p.add_argument("mode", choices=["llm", "image"])
    p.add_argument("--seqlens", default="4096,8192,16384,32768,65536")
    p.add_argument("--groups", default="2,8")
    p.add_argument("--batch", type=int, default=1)
    p.add_argument("--heads", type=int, default=32)
    p.add_argument("--kv-heads", type=int, default=8)
    p.add_argument("--image-heads", type=int, default=24)
    p.add_argument("--warmup", type=int, default=10)
    p.add_argument("--iters", type=int, default=50)
    p.add_argument("--out", help="optional JSON output path")
    args = p.parse_args()
    args.seqlens = [int(x) for x in args.seqlens.split(",")]
    args.groups = [int(x) for x in args.groups.split(",")]
    print(f"device: {torch.cuda.get_device_name()}  torch {torch.__version__}")
    rows = bench_llm(args) if args.mode == "llm" else bench_image(args)
    if args.out:
        json.dump({"device": torch.cuda.get_device_name(), "torch": torch.__version__,
                   "args": {k: v for k, v in vars(args).items() if k != "out"}, "rows": rows},
                  open(args.out, "w"), indent=1)


if __name__ == "__main__":
    main()
