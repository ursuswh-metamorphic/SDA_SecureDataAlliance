#!/bin/bash
set -euo pipefail

ROOT=/workspace/SDA_SecureDataAlliance
FEDE="$ROOT/FedE"
OUT="$ROOT/artifacts/training/b1_central_pilot_seed13"
EVAL_OUT="$ROOT/artifacts/eval/b1_central_pilot_val_seed13"
LOG="$OUT/job.log"
DONE="$OUT/job.done"

mkdir -p "$OUT" "$EVAL_OUT"
rm -f "$DONE"
cd "$FEDE"

{
  echo "run_name=b1_central_pilot_seed13"
  echo "model=BAAI/bge-base-en-v1.5"
  echo "pooling=cls"
  echo "num_clients=1"
  echo "use_lora=0"
  echo "dp_enabled=0"
  echo "kd_weight=0"
  echo "rounds=15"
  echo "steps_per_round=50"
  echo "batch_size=16"
  echo "learning_rate=1e-5"
  echo "seed=13"
  sha256sum \
    main_lora.py \
    flgo/algorithm/fedrag_lora.py \
    flgo/benchmark/fedrag_classification/core.py \
    flgo/benchmark/fedrag_classification/config.py \
    selected_data_clean.json
} >"$OUT/manifest.txt"

set +e
env \
  NUM_CLIENTS=1 \
  USE_LORA=0 \
  USE_QLORA=0 \
  DP_ENABLED=0 \
  KD_WEIGHT=0 \
  POOLING=cls \
  NUM_ROUNDS=15 \
  NUM_STEPS=50 \
  BATCH_SIZE=16 \
  GPU=0 \
  SEED=13 \
  DATA_SEED=13 \
  PYTHONUNBUFFERED=1 \
  /workspace/sda-venv/bin/python -u main_lora.py >"$LOG" 2>&1
train_rc=$?
set -e

if [[ "$train_rc" -ne 0 ]]; then
  echo "TRAIN_EXIT_CODE=$train_rc" >>"$LOG"
  echo "$train_rc" >"$DONE"
  exit "$train_rc"
fi

checkpoint=$(ls -1t x-lora_*.bin | head -1)
mv "$checkpoint" "$OUT/checkpoint.bin"
sha256sum "$OUT/checkpoint.bin" >"$OUT/checkpoint.sha256"

set +e
/workspace/sda-venv/bin/python -u eval_clean.py \
  --corpus ../artifacts/data/corpus_v1.jsonl \
  --queries ../artifacts/data/queries_val_v1.jsonl \
  --qrels ../artifacts/data/qrels_val_v1.tsv \
  --model BAAI/bge-base-en-v1.5 \
  --checkpoint "$OUT/checkpoint.bin" \
  --pooling cls \
  --device cuda \
  --batch-size 64 \
  --top-k 100 \
  --seed 13 \
  --run-name b1_central_pilot_val_seed13 \
  --output "$EVAL_OUT" >>"$LOG" 2>&1
eval_rc=$?
set -e

echo "EVAL_EXIT_CODE=$eval_rc" >>"$LOG"
echo "$eval_rc" >"$DONE"
exit "$eval_rc"
