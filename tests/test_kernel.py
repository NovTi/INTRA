"""Parity of every INTRA kernel against the fp32 dense-mask reference.

    python tests/test_kernel.py            # exits non-zero on any mismatch
"""
import sys

import torch

from intra_attn import intra_image_attn, intra_llm_attn
from intra_attn.reference import (TEXT_LEN, image_gather0_mask, image_scatter_mask, llm_gather1_mask,
                                  llm_scatter_mask, llm_window_mask, masked_attention)

ATOL = 2e-2  # bf16 inputs/outputs; observed errors are ~1e-3
torch.manual_seed(0)


def run(name, fn, mask, B, S, H, Hkv):
    q = torch.randn(B, S, H, 128, device="cuda", dtype=torch.bfloat16)
    k = torch.randn(B, S, Hkv, 128, device="cuda", dtype=torch.bfloat16)
    v = torch.randn(B, S, Hkv, 128, device="cuda", dtype=torch.bfloat16)
    with torch.no_grad():
        out = fn(q, k, v)
        ref = masked_attention(q, k, v, mask)
    err = (out.float() - ref.float()).abs().max().item()
    ok = bool(torch.isfinite(out).all()) and err < ATOL
    print(f"{'PASS' if ok else 'FAIL'}  {name:42s} B={B} S={S:6d} H={H}/{Hkv}  max|err|={err:.2e}")
    return ok


def main():
    dev = "cuda"
    ok = True
    for S in (1000, 2048, 3000, 4096, 8192, 16384):
        for B, H, Hkv in ((1, 32, 8), (2, 32, 32)):
            for G in (2, 3, 8):
                ok &= run(f"llm scatter G={G}", lambda q, k, v: intra_llm_attn(q, k, v, "scatter", group_num=G),
                          llm_scatter_mask(S, G, dev), B, S, H, Hkv)
            G = 8
            if S % 64 == 0:
                ok &= run(f"llm gather1 G={G}", lambda q, k, v: intra_llm_attn(q, k, v, "gather1", group_num=G),
                          llm_gather1_mask(S, G, dev), B, S, H, Hkv)
            for W in (576, 1000):
                ok &= run(f"llm window W={W}", lambda q, k, v: intra_llm_attn(q, k, v, "window", window=W),
                          llm_window_mask(S, W, device=dev), B, S, H, Hkv)
    for width, H in ((64, 24), (128, 8)):
        S = TEXT_LEN + width * width
        for B in (1, 2):
            ok &= run(f"image scatter W={width}", lambda q, k, v: intra_image_attn(q, k, v, "scatter", width),
                      image_scatter_mask(width, device=dev), B, S, H, H)
            ok &= run(f"image gather0 W={width}", lambda q, k, v: intra_image_attn(q, k, v, "gather0", width),
                      image_gather0_mask(width, device=dev), B, S, H, H)
    print("ALL PASS" if ok else "SOME FAILED")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
