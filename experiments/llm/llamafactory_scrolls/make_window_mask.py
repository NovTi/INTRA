#!/usr/bin/env python3
"""Build the additive sliding-window mask loaded by model/llama3_student.py ($INTRA_WINDOW_MASK).

Query i sees keys j with  max(chunk * (i // chunk) - back, 0) <= j <= i,  i.e. a causal window that
starts `back` tokens before the query's 64-aligned chunk (up to back + chunk = 576 keys). Visible
entries are 0, the rest -inf; shape (1, 1, seq_len, seq_len), bfloat16. The student slices it to the
current sequence length, so seq_len must be >= the longest training sequence.

  python make_window_mask.py --out mask4.pt                 # defaults used for the paper runs
  python make_window_mask.py --selftest
"""
import argparse

import torch


def window_mask(seq_len, back=512, chunk=64, dtype=torch.bfloat16):
    i = torch.arange(seq_len).view(-1, 1)
    j = torch.arange(seq_len).view(1, -1)
    start = (i // chunk * chunk - back).clamp(min=0)
    vis = (j >= start) & (j <= i)
    zero = torch.zeros((), dtype=dtype)
    ninf = torch.full((), float("-inf"), dtype=dtype)
    return torch.where(vis, zero, ninf).view(1, 1, seq_len, seq_len)


def _loop_reference(seq_len, back, chunk):
    """Element-wise construction, as originally written."""
    mask = torch.full((1, 1, seq_len, seq_len), float("-inf"), dtype=torch.bfloat16)
    for idx in range(seq_len):
        chunk_start = max(idx // chunk * chunk - back, 0)
        for k in range(chunk_start, idx + 1):
            mask[:, :, idx, k] = 0
    return mask


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--seq_len", type=int, default=16400)
    ap.add_argument("--back", type=int, default=512)
    ap.add_argument("--chunk", type=int, default=64)
    ap.add_argument("--out", default="mask4.pt")
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args()
    if a.selftest:
        for S, b, c in [(700, 512, 64), (300, 64, 16), (129, 32, 8)]:
            assert torch.equal(window_mask(S, b, c), _loop_reference(S, b, c)), (S, b, c)
        print("selftest PASS")
    else:
        torch.save(window_mask(a.seq_len, a.back, a.chunk), a.out)
        print(f"wrote {a.out}")
