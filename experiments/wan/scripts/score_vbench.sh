#!/usr/bin/env bash
# Score both arms on the 13 VBench dimensions of Table 18, one single-process job per dimension
# and GPU, then print the table. Run with the Python of a separate VBench environment:
#   VBENCH_PY  python with vbench installed (default: python)
#   PROMPTS    VBench_full_info.json
#   VBENCH_CACHE_DIR  where VBench keeps / downloads its model checkpoints
set -euo pipefail
W=$(cd "$(dirname "$0")/.." && pwd)
PY=${VBENCH_PY:-python}
OUT_DIR=${OUT_DIR:-outputs}
PROMPTS=${PROMPTS:?set PROMPTS to VBench_full_info.json}
NGPU=${NGPU:-8}
VID=$OUT_DIR/vbench_videos
RES=$OUT_DIR/vbench_results
DIMS=${DIMS:-"subject_consistency background_consistency aesthetic_quality imaging_quality
temporal_flickering motion_smoothness dynamic_degree overall_consistency appearance_style
temporal_style object_class multiple_objects spatial_relationship"}
mkdir -p "$RES" "$OUT_DIR/logs"

# ---- compatibility fixes for running VBench on numpy 2 / transformers 5 (idempotent) ----------
VB=$("$PY" -c "import os, vbench; print(os.path.dirname(vbench.__file__))")
# numpy 2 removed numpy.lib.function_base; the imported `disp` is unused (human_action).
{ grep -rl "^from numpy.lib.function_base import" "$VB" || true; } | while read -r f; do
  sed -i 's/^from numpy\.lib\.function_base import .*/pass  # numpy>=2: unused import removed/' "$f"
done
# transformers 5 moved apply_chunking_to_forward / prune_linear_layer to pytorch_utils and removed
# find_pruneable_heads_and_indices (only used for head pruning, never called by VBench) (scene).
"$PY" - "$VB/third_party/tag2Text/med.py" <<'PY'
import re, sys
p = sys.argv[1]
s = open(p).read()
m = re.search(r"from transformers\.modeling_utils import \([^)]*\)\n", s)
if m and "INTRA-PATCH" not in s:
    fix = '''# INTRA-PATCH: per-name imports that work on transformers 4.x and 5.x
from transformers.modeling_utils import PreTrainedModel
try:
    from transformers.pytorch_utils import apply_chunking_to_forward, prune_linear_layer
except ImportError:
    from transformers.modeling_utils import apply_chunking_to_forward, prune_linear_layer
try:
    from transformers.pytorch_utils import find_pruneable_heads_and_indices
except ImportError:
    import torch as _torch

    def find_pruneable_heads_and_indices(heads, n_heads, head_size, already_pruned_heads):
        mask = _torch.ones(n_heads, head_size)
        heads = set(heads) - already_pruned_heads
        for head in heads:
            head = head - sum(1 if h < head else 0 for h in already_pruned_heads)
            mask[head] = 0
        mask = mask.view(-1).contiguous().eq(1)
        index = _torch.arange(len(mask))[mask].long()
        return heads, index
'''
    open(p, "w").write(s[:m.start()] + fix + s[m.end():])
PY
# object_class / multiple_objects / color / spatial_relationship use GRiT, which needs detectron2:
#   $PY -m pip install --no-build-isolation 'git+https://github.com/facebookresearch/detectron2.git'
"$PY" -c "import detectron2" 2>/dev/null || echo "warning: detectron2 missing; GRiT dimensions will fail"

# ---- score ------------------------------------------------------------------------------------
for arm in dense intra; do
  g=0
  for d in $DIMS; do
    [ -f "$RES/${arm}_${d}_eval_results.json" ] && continue
    CUDA_VISIBLE_DEVICES=$g "$PY" "$W/vbench_score.py" score --videos "$VID/$arm" --arm "$arm" --dim "$d" \
      --info "$PROMPTS" --results "$RES" > "$OUT_DIR/logs/vbench_${arm}_${d}.log" 2>&1 &
    g=$(( (g + 1) % NGPU ))
    [ "$g" = 0 ] && wait
  done
  wait
done
"$PY" "$W/vbench_score.py" table --results "$RES"
