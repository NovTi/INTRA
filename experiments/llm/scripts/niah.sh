#!/usr/bin/env bash
# Teacher-forced NIAH for one arm: PG19 haystack, 9 depths x 10 trials, lengths 4K/8K/16K/32K
# (one length per GPU, GPUs 0-3). Same arm spec as train_arm.sh; LORA=NONE evaluates without an
# adapter. Output: $OUT_DIR/niah/${ARM}_L<len>.json, then a summary table.
#
#   ARM=g2s4 NGROUPS=2 NOFF=2 PATTERN=SWWWWWWW bash scripts/niah.sh
source "$(dirname "${BASH_SOURCE[0]}")/common.sh"
arm_flags
lora_args
mkdir -p "$OUT_DIR/niah"
cd "$HERE"
gpu=0
for L in 4096 8192 16384 32768; do
  CUDA_VISIBLE_DEVICES=$gpu $PYTHON niah_eval.py --model "$MODEL" $NIAH_FLAGS $LORA_ARGS \
    --lengths "$L" --depths 0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9 --trials 10 \
    --haystack "$HAYSTACK" --out "$OUT_DIR/niah/${ARM}_L$L.json" \
    > "$OUT_DIR/niah/${ARM}_L$L.log" 2>&1 &
  gpu=$((gpu + 1))
done
wait
$PYTHON niah_table.py --dir "$OUT_DIR/niah" "$ARM"
