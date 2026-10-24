#!/usr/bin/env bash
# Shared settings for scripts/*.sh. Sourced, not run. Every path is overridable from the environment.
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"          # experiments/llm
MODEL=${MODEL:-meta-llama/Llama-3.1-8B-Instruct}                 # or Qwen/Qwen3-8B
DATA_DIR=${DATA_DIR:-$HERE/data}
OUT_DIR=${OUT_DIR:-$HERE/outputs}
SCROLLS_DIR=${SCROLLS_DIR:-$DATA_DIR/scrolls}                    # <task>.{train,validation}.jsonl
EVAL_FILE=${EVAL_FILE:-$DATA_DIR/scrolls6_n200.jsonl}
CORPUS=${CORPUS:-$DATA_DIR/mixgen_16k_$(basename "$MODEL")}      # tokenizer-specific
HAYSTACK=${HAYSTACK:-$DATA_DIR/pg19/*.parquet}                   # glob, expanded by niah_eval.py
NGPU=${NGPU:-8}
PYTHON=${PYTHON:-python}

export TORCHINDUCTOR_COMPILE_THREADS=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# arm_flags: derive the attention flags for training, NIAH and SCROLLS evaluation from ONE arm spec,
# so the three stages always see the same mask.
#   ARM      run name (required)
#   STUDENT  intra (default) | dense | moba | nsa | minference | xattention (the last two are
#            training-free and need the transformers-4.46 environment, see README)
#   intra only: NGROUPS, NOFF (stripe offsets; 1 = no stripe), PATTERN (S/W/D letters), WINDOW (575)
#   LORA     adapter path (default $OUT_DIR/runs/$ARM/lora_final.pt); NONE = no adapter
arm_flags() {
  : "${ARM:?set ARM}"
  STUDENT=${STUDENT:-intra}
  LORA=${LORA:-$OUT_DIR/runs/$ARM/lora_final.pt}
  case "$STUDENT" in
    intra)
      : "${NGROUPS:?set NGROUPS}" "${NOFF:?set NOFF}" "${PATTERN:?set PATTERN}"
      WINDOW=${WINDOW:-575}
      # llama_intra.py silently drops letters other than S/W/D, which would change the cycle length.
      case "$PATTERN" in *[!SWD]*) echo "PATTERN may only contain S, W, D (got $PATTERN)" >&2; exit 1;; esac
      local stripe="" estripe=""
      if [ "$NOFF" -gt 1 ]; then
        stripe="--scatter_stripe 1 --scatter_n_offsets $NOFF"
        estripe="--intra_stripe 1 --intra_n_offsets $NOFF"
      fi
      TRAIN_FLAGS="--student_attn intra --num_groups $NGROUPS --window $WINDOW --scatter_blocks 256 --n_sink 64 --layer_pattern $PATTERN $stripe --scatter_impl flex --compile_flex 1"
      NIAH_FLAGS="--mode sparse --scatter_blocks 256 --n_sink 64 --window $WINDOW --layer_pattern $PATTERN --num_groups $NGROUPS $stripe --scatter_impl flex"
      EVAL_FLAGS="--backend intra_llama --intra_block 256 --intra_sink 64 --intra_window_flex $WINDOW --layer_pattern $PATTERN --intra_groups $NGROUPS $estripe --intra_impl flex"
      ;;
    dense)
      TRAIN_FLAGS="--student_attn dense"
      NIAH_FLAGS="--mode dense"
      EVAL_FLAGS="--backend dense"
      ;;
    moba)
      TRAIN_FLAGS="--student_attn moba --moba_density 0.125 --moba_chunk 2048"
      NIAH_FLAGS="--mode moba --moba_density 0.125 --moba_chunk 2048"
      EVAL_FLAGS="--backend moba --moba_density 0.125 --moba_chunk 2048"
      ;;
    nsa)
      # nominal density 0.1438 measures ~0.126 (nsa_attn.calibrate_density)
      TRAIN_FLAGS="--student_attn nsa --nsa_density 0.1438 --nsa_sel_block 64"
      NIAH_FLAGS="--mode nsa --nsa_density 0.1438"
      EVAL_FLAGS="--backend nsa --nsa_density 0.1438 --nsa_sel_block 64"
      ;;
    minference)
      LORA=NONE; TRAIN_FLAGS=""
      NIAH_FLAGS="--mode minference"
      EVAL_FLAGS="--backend minference"
      ;;
    xattention)
      LORA=NONE; TRAIN_FLAGS=""
      NIAH_FLAGS="--mode xattention --xattn_stride 16"
      EVAL_FLAGS="--backend xattention --xattn_stride 16"
      ;;
    *) echo "unknown STUDENT=$STUDENT" >&2; exit 1 ;;
  esac
}

# lora_args: evaluation-side adapter flags (after arm_flags).
lora_args() {
  LORA_ARGS=""
  if [ "$LORA" != NONE ]; then
    [ -f "$LORA" ] || { echo "no adapter at $LORA (set LORA=NONE to evaluate without one)" >&2; exit 1; }
    LORA_ARGS="--lora $LORA --lora_rank 16"
  fi
}
