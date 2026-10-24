#!/usr/bin/env bash
# Generate the VBench suite (944 prompts x 5 samples, 480x832, 49 frames) for the dense model and
# for INTRA + the distilled adapter. One shard per GPU; resumable.
#   PROMPTS = VBench_full_info.json from the VBench repo
#   LORA    = adapter from scripts/train.sh
#   ARMS    = "dense intra" (default) or one of them
set -euo pipefail
W=$(cd "$(dirname "$0")/.." && pwd)
MODEL=${MODEL:-Wan-AI/Wan2.2-TI2V-5B-Diffusers}
OUT_DIR=${OUT_DIR:-outputs}
PROMPTS=${PROMPTS:?set PROMPTS to VBench_full_info.json}
LORA=${LORA:-$OUT_DIR/intra_r32/lora_final.pt}
NGPU=${NGPU:-8}
VID=$OUT_DIR/vbench_videos
mkdir -p "$OUT_DIR/logs"

for arm in ${ARMS:-dense intra}; do
  extra=()
  if [ "$arm" = intra ]; then
    [ -f "$LORA" ] || { echo "no adapter at $LORA"; exit 1; }
    extra=(--lora "$LORA" --lora_rank 32 --ffn_lora 1
           --scatter_a 2,4 --scatter_b 4,2 --tile auto --tile_target 8 --cone_wt 4)
  fi
  for i in $(seq 0 $((NGPU - 1))); do
    CUDA_VISIBLE_DEVICES=$i python "$W/gen_vbench.py" --mode "$arm" --model "$MODEL" \
      --prompts "$PROMPTS" --out "$VID/$arm" --meta_dir "$VID/${arm}_meta" \
      --frames 49 --height 480 --width 832 --steps 50 --guidance 5.0 --fps 16 --samples 5 \
      --shard "$i" --nshard "$NGPU" ${extra[@]+"${extra[@]}"} > "$OUT_DIR/logs/gen_${arm}_$i.log" 2>&1 &
  done
  wait
  # VBench takes the extension from the first directory entry: anything but .mp4 breaks matching.
  n_mp4=$(find "$VID/$arm" -maxdepth 1 -name '*.mp4' | wc -l)
  n_all=$(find "$VID/$arm" -maxdepth 1 -mindepth 1 | wc -l)
  echo "$arm: $n_mp4/4720 mp4, $n_all entries"
  [ "$n_mp4" = "$n_all" ] || { echo "non-mp4 entries in $VID/$arm"; exit 1; }
done
