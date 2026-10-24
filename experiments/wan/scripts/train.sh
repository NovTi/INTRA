#!/usr/bin/env bash
# DoRA self-distillation with the recipe of the paper's VBench checkpoint: rank 32 (attn1 + FFN),
# 20000 steps, 8 GPUs x batch 1, Prodigy lr 1, unconditional-branch probability 0.5,
# INTRA scatter (2,4)/(4,2), auto Gather tiles, cone W_t = 4, blocks 0 and 29 dense.
set -euo pipefail
W=$(cd "$(dirname "$0")/.." && pwd)
MODEL=${MODEL:-Wan-AI/Wan2.2-TI2V-5B-Diffusers}
OUT_DIR=${OUT_DIR:-outputs}
NGPU=${NGPU:-8}
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

torchrun --standalone --nproc_per_node "$NGPU" "$W/train.py" \
  --model "$MODEL" --cache "$OUT_DIR/teacher_clips" --out "$OUT_DIR/intra_r32" \
  --steps 20000 --grad_accum 1 --lora_rank 32 --ffn_lora 1 --flow_shift 5.0 --grad_ckpt 1 \
  --p_uncond 0.5 --save_every 1000 --log_every 100 \
  --scatter_a 2,4 --scatter_b 4,2 --tile auto --tile_target 8 --cone_wt 4
