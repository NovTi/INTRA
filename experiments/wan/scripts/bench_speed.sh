#!/usr/bin/env bash
# Table 19: dense vs INTRA timing on one GPU (paper: A100-SXM4-80GB).
set -euo pipefail
W=$(cd "$(dirname "$0")/.." && pwd)
MODEL=${MODEL:-Wan-AI/Wan2.2-TI2V-5B-Diffusers}
OUT=${OUT_DIR:-outputs}/speed
B() { python "$W/bench_speed.py" --model "$MODEL" --out "$OUT" "$@"; }

B parity --grids 480p_49f,720p_121f              # fast path == reference == fp32 oracle
B plumbing                                       # processor == stock diffusers (dense core)
B attn                                           # self-attention rows
B dit --iters 10 --warmup 3                      # standalone DiT forward rows
for arm in dense intra; do                        # end-to-end rows
  for g in 480p_49f 720p_121f; do
    B e2e --arm "$arm" --grids "$g"
  done
done
