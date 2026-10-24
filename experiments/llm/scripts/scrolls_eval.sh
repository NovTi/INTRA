#!/usr/bin/env bash
# SCROLLS (6 tasks x 200 validation records) for one arm, sharded over NGPU GPUs, then scored with
# max-over-references. Same arm spec as train_arm.sh. Output: $OUT_DIR/scrolls/${ARM}_shard*.jsonl
#
#   ARM=g2s4 NGROUPS=2 NOFF=2 PATTERN=SWWWWWWW bash scripts/scrolls_eval.sh
#   ARM=minference STUDENT=minference PYTHON=/path/to/transformers446-env/bin/python bash scripts/scrolls_eval.sh
source "$(dirname "${BASH_SOURCE[0]}")/common.sh"
arm_flags
lora_args
mkdir -p "$OUT_DIR/scrolls"
cd "$HERE"
for i in $(seq 0 $((NGPU - 1))); do
  CUDA_VISIBLE_DEVICES=$i $PYTHON run_eval.py --data "$EVAL_FILE" --model "$MODEL" \
    $EVAL_FLAGS $LORA_ARGS --shard "$i" --nshard "$NGPU" \
    --out "$OUT_DIR/scrolls/${ARM}_shard$i.jsonl" > "$OUT_DIR/scrolls/${ARM}_shard$i.log" 2>&1 &
done
wait
$PYTHON score.py --data "$EVAL_FILE" --refs "$SCROLLS_DIR" --arm "$ARM=$OUT_DIR/scrolls/${ARM}_shard*.jsonl"
