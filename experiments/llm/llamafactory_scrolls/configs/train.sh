#!/usr/bin/env bash
# Run one distillation config on a single GPU (as the original runs did) from the LLaMA-Factory root.
#   INTRA_DISTILL_DATA=/path/to/hf_dataset INTRA_WINDOW_MASK=/path/to/mask4.pt \
#     bash configs/train.sh configs/distill_scrolls_rank8.yaml
set -euo pipefail
: "${INTRA_DISTILL_DATA:?set INTRA_DISTILL_DATA to a local HF dataset dir (alpaca columns)}"
: "${INTRA_WINDOW_MASK:?set INTRA_WINDOW_MASK to the mask from make_window_mask.py}"
export INTRA_DISTILL_DATA INTRA_WINDOW_MASK
CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0} llamafactory-cli train "${1:?config yaml}"
