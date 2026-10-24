#!/usr/bin/env bash
# Teacher data: 600 templated prompts; dense Wan2.2-TI2V-5B clips for prompts 0-499
# (480x832, 49 frames, 50 steps, CFG 5) plus the empty-prompt embedding.
# Prompts 500-599 are not generated (held out).
set -euo pipefail
W=$(cd "$(dirname "$0")/.." && pwd)
MODEL=${MODEL:-Wan-AI/Wan2.2-TI2V-5B-Diffusers}
OUT_DIR=${OUT_DIR:-outputs}
NGPU=${NGPU:-8}

torchrun --standalone --nproc_per_node "$NGPU" "$W/selfgen.py" \
  --model "$MODEL" --out "$OUT_DIR/teacher_clips" \
  --n_prompts 600 --prompt_seed 1234 --gid_start 0 --gid_end 500 \
  --frames 49 --height 480 --width 832 --steps 50 --guidance 5.0
echo "clips: $(ls "$OUT_DIR"/teacher_clips/clip_*.pt | wc -l) (expected 500)"
