#!/usr/bin/env bash
# LoRA self-distillation of INTRA (Scatter / Gather 0, switching every two layers) into FLUX.1-dev
# at 1024 x 1024, with the archived training configuration of the paper's FLUX run.
#
#   CLEAR_DIR=/path/to/CLEAR DATA_DIR=/path/to/t2i_1024 bash scripts/train.sh
set -euo pipefail
cd "$(dirname "$0")/.."

MODEL=${MODEL:-black-forest-labs/FLUX.1-dev}
CLEAR_DIR=${CLEAR_DIR:?set CLEAR_DIR to a clone of https://github.com/Huage001/CLEAR (provides dataset.py)}
DATA_DIR=${DATA_DIR:?set DATA_DIR to the CLEAR 10K training set with cached prompt embeddings and latents}
OUTPUT_DIR=${OUTPUT_DIR:-output/intra_flux}
REPORT_TO=${REPORT_TO:-wandb}
export PYTHONPATH="${CLEAR_DIR}${PYTHONPATH:+:$PYTHONPATH}"

# --instance_prompt is required by the argument parser but unused (prompts come from the cached embeddings).
accelerate launch train_distill.py \
  --pretrained_model_name_or_path="$MODEL" \
  --instance_data_dir="$DATA_DIR" \
  --output_dir="$OUTPUT_DIR" \
  --mixed_precision="bf16" \
  --instance_prompt="a photo of dog" \
  --resolution=1024 \
  --train_batch_size=1 \
  --guidance_scale=1 \
  --gradient_accumulation_steps=8 \
  --optimizer="prodigy" \
  --learning_rate=1. \
  --report_to="$REPORT_TO" \
  --lr_scheduler="constant" \
  --lr_warmup_steps=0 \
  --max_train_steps=50000 \
  --checkpointing_steps=1000 \
  --validation_prompt="A photo of dog in a bucket" \
  --validation_epochs=1 \
  --gradient_checkpointing \
  --seed="42" \
  --dataloader_num_workers=2 \
  --exp_name=intra_flux
