#!/bin/bash
set -euo pipefail

ROOT=/workspace/SDA_SecureDataAlliance
FEDE="$ROOT/FedE"
OUT="$ROOT/artifacts/eval/b1_central_smoke_cls"
LOG="$OUT/job.log"
DONE="$OUT/job.done"

mkdir -p "$OUT"
rm -f "$DONE"
cd "$FEDE"

set +e
/workspace/sda-venv/bin/python -u eval_clean.py \
  --corpus ../artifacts/data/corpus_v1.jsonl \
  --queries ../artifacts/data/queries_val_v1.jsonl \
  --qrels ../artifacts/data/qrels_val_v1.tsv \
  --model BAAI/bge-base-en-v1.5 \
  --checkpoint x-lora_2026-07-29_13-08-00.bin \
  --pooling cls \
  --device cuda \
  --batch-size 64 \
  --top-k 100 \
  --seed 13 \
  --run-name b1_central_smoke_cls \
  --output "$OUT" >"$LOG" 2>&1
rc=$?
set -e

echo "EXIT_CODE=$rc" >>"$LOG"
echo "$rc" >"$DONE"
exit "$rc"
