#!/usr/bin/env bash
# Data for the LoRA-adaptation / NIAH experiments. Idempotent; rerun with MODEL=Qwen/Qwen3-8B to pack
# the Qwen3-tokenised corpus (downloads and the eval file are reused).
#   1. official SCROLLS train + validation splits (tau/scrolls)      -> $SCROLLS_DIR
#   2. PG19 test split parquet (emozilla/pg19-test), NIAH haystack    -> $DATA_DIR/pg19
#   3. SCROLLS eval file, 6 tasks x 200 records (seed 1234)           -> $EVAL_FILE
#   4. mixgen distillation corpus, 400 x 16384 tokens (60/25/15)      -> $CORPUS
source "$(dirname "${BASH_SOURCE[0]}")/common.sh"
mkdir -p "$SCROLLS_DIR" "$DATA_DIR/pg19"
cd "$HERE"

BASE=https://huggingface.co/datasets/tau/scrolls/resolve/main
for t in gov_report summ_screen_fd qmsum qasper quality contract_nli; do
  if [ -s "$SCROLLS_DIR/$t.train.jsonl" ] && [ -s "$SCROLLS_DIR/$t.validation.jsonl" ]; then continue; fi
  tmp=$(mktemp -d)
  curl -sSL --retry 3 --retry-delay 5 -o "$tmp/$t.zip" "$BASE/$t.zip"
  unzip -o -q "$tmp/$t.zip" -d "$tmp/x"
  for split in train validation; do
    f=$(find "$tmp/x" -name "$split.jsonl*" | head -1)
    [ -n "$f" ] || { echo "no $split.jsonl in $t.zip" >&2; exit 1; }
    case "$f" in *.gz) gunzip -c "$f" > "$SCROLLS_DIR/$t.$split.jsonl" ;; *) mv "$f" "$SCROLLS_DIR/$t.$split.jsonl" ;; esac
  done
  rm -rf "$tmp"
  echo "$t: $(wc -l < "$SCROLLS_DIR/$t.train.jsonl") train / $(wc -l < "$SCROLLS_DIR/$t.validation.jsonl") validation rows"
done

if ! ls "$DATA_DIR"/pg19/*.parquet > /dev/null 2>&1; then
  $PYTHON - "$DATA_DIR/pg19" <<'PY'
import glob, os, shutil, sys
from huggingface_hub import snapshot_download
src = snapshot_download("emozilla/pg19-test", repo_type="dataset", allow_patterns=["*.parquet"])
for f in sorted(glob.glob(os.path.join(src, "**", "*.parquet"), recursive=True)):
    shutil.copy(f, sys.argv[1])
    print("pg19:", os.path.basename(f))
PY
fi

[ -s "$EVAL_FILE" ] || $PYTHON build_data.py --src "$SCROLLS_DIR" --out "$EVAL_FILE" --n 200 --seed 1234

[ -f "$CORPUS/meta.json" ] || $PYTHON pack_scrolls_mix.py --tokenizer "$MODEL" \
  --scrolls_train "$SCROLLS_DIR" --eval_file "$EVAL_FILE" \
  --seq_len 16384 --n_blocks 400 --frac_task 0.60 --frac_retrieval 0.25 --out "$CORPUS"
