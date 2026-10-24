#!/usr/bin/env bash
# Train one LoRA adapter by self-distillation (paper recipe: 16K-token mixgen blocks, 1200 steps,
# rank-16 LoRA on q/k/v/o + gate/up/down, KL T=2 alpha=0.9 + 0.1 CE, torchrun on NGPU GPUs).
#
#   ARM=g2s4 NGROUPS=2 NOFF=2 PATTERN=SWWWWWWW bash scripts/train_arm.sh     # paper INTRA arm
#   ARM=dense STUDENT=dense bash scripts/train_arm.sh                        # dense + LoRA control
# Arm spec: see arm_flags in scripts/common.sh. Output: $OUT_DIR/runs/$ARM/lora_final.pt
source "$(dirname "${BASH_SOURCE[0]}")/common.sh"
arm_flags
[ -n "$TRAIN_FLAGS" ] || { echo "STUDENT=$STUDENT is training-free" >&2; exit 1; }
STEPS=${STEPS:-1200}
RUN=$OUT_DIR/runs/$ARM
if [ -f "$RUN/lora_final.pt" ]; then echo "$ARM: $RUN/lora_final.pt exists, skipping"; exit 0; fi
[ -f "$CORPUS/meta.json" ] || { echo "no corpus at $CORPUS (run scripts/prepare_data.sh)" >&2; exit 1; }

cd "$HERE"
torchrun --standalone --nproc_per_node "$NGPU" train_llama_distill.py \
  --model "$MODEL" --data "$CORPUS" --seq_len 16384 --lora_rank 16 \
  --kl_T 2.0 --alpha 0.9 --ce_weight 0.1 --chunk 2048 --log_every 50 --save_every 400 \
  $TRAIN_FLAGS --steps "$STEPS" --out "$RUN"
