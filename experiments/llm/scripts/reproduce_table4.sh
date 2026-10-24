#!/usr/bin/env bash
# Every LoRA arm behind the LLaMA-3.1-8B NIAH numbers in paper Table 4a, Table 5, Table 15 and
# Table 16: train (8 GPUs) then NIAH (4 GPUs). Run scripts/prepare_data.sh first.
# INTRA arms take about 1 h each on 8x A100; MoBA and NSA students materialise additive masks and
# are several times slower.
set -euo pipefail
S="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

arm() {  # arm NAME STUDENT [NGROUPS NOFF WINDOW PATTERN]
  export ARM=$1 STUDENT=$2 NGROUPS=${3:-} NOFF=${4:-} WINDOW=${5:-575} PATTERN=${6:-}
  bash "$S/train_arm.sh"
  bash "$S/niah.sh"
}

# Table 4a: Dense + LoRA / MoBA / NSA / INTRA (INTRA = mean of three independent runs; the trainer
# has no seed flag, so g2s4, g2s4b, g2s4c differ only in LoRA init and data order)
arm dense  dense
arm moba   moba
arm nsa    nsa
arm g2s4   intra 2 2 575 SWWWWWWW
arm g2s4b  intra 2 2 575 SWWWWWWW
arm g2s4c  intra 2 2 575 SWWWWWWW
# Table 5: the same four layers as sliding window / INTRA (g2s4) / dense
arm wonly575  intra 2 1 575 WWWWWWWW
arm dense4    intra 2 1 575 DWWWWWWW
# Table 15: all-window stacks (with wonly575)
arm wonly2048 intra 2 1 2048 WWWWWWWW
arm wonly4096 intra 2 1 4096 WWWWWWWW
# Table 16: sparse attention in 16 layers (with wonly2048 = "wider windows", g2s4, dense)
arm baseb     intra 8 1 575 SSWW

source "$S/common.sh"
cd "$HERE"
$PYTHON niah_table.py --dir "$OUT_DIR/niah" dense moba nsa g2s4 g2s4b g2s4c \
  wonly575 dense4 wonly2048 wonly4096 baseb
