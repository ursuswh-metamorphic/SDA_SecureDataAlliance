#!/bin/bash
set -euo pipefail

ROOT=/workspace/SDA_SecureDataAlliance
FEDE="$ROOT/FedE"
SUMMARY="$ROOT/artifacts/training/b4_fedprox_mu_sweep_seed13/sweep_summary.json"
PYTHON=/workspace/sda-venv/bin/python

if [[ -n "${FEDPROX_MU:-}" ]]; then
  MU="$FEDPROX_MU"
else
  MU=$("$PYTHON" -c \
    "import json; print(json.load(open('$SUMMARY', encoding='utf-8'))['best_mu'])")
fi

TAG="${MU//./p}"
OUT="$ROOT/artifacts/training/b4_fedprox_full_pilot_mu${TAG}_seed13"
EVAL_OUT="$ROOT/artifacts/eval/b4_fedprox_full_pilot_mu${TAG}_val_seed13"
WORK="$OUT/work"
LOG="$OUT/job.log"
DONE="$OUT/job.done"

mkdir -p "$OUT" "$EVAL_OUT" "$WORK"
rm -f "$DONE"
cd "$WORK"

{
  echo "run_name=b4_fedprox_full_pilot_mu${MU}_seed13"
  echo "selection_source=$SUMMARY"
  echo "protocol=5 company-exclusive clients; FedProx; full fine-tune"
  echo "fedprox_mu=$MU"
  echo "model=BAAI/bge-base-en-v1.5"
  echo "pooling=cls"
  echo "num_clients=5"
  echo "use_lora=0"
  echo "dp_enabled=0"
  echo "kd_weight=0"
  echo "rounds=15"
  echo "steps_per_round=50"
  echo "batch_size=16"
  echo "learning_rate=1e-5"
  echo "seed=13"
  sha256sum \
    "$FEDE/main_lora.py" \
    "$FEDE/flgo/algorithm/fedrag_fedprox.py" \
    "$FEDE/flgo/algorithm/fedprox_utils.py" \
    "$FEDE/selected_data_clean.json" \
    "$SUMMARY"
} >"$OUT/manifest.txt"

exec >>"$LOG" 2>&1
echo "[B4-full] started mu=$MU at $(date -u +%FT%TZ)"

env \
  PYTHONPATH="$FEDE" \
  TRAIN_DATA="$FEDE/selected_data_clean.json" \
  TASK_PATH="$OUT/task" \
  ALGORITHM=fedprox \
  FEDPROX_MU="$MU" \
  NUM_CLIENTS=5 \
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
  HF_HUB_OFFLINE=1 \
  TRANSFORMERS_OFFLINE=1 \
  PYTHONUNBUFFERED=1 \
  "$PYTHON" -u "$FEDE/main_lora.py"

checkpoint=$(ls -1t x-lora_*.bin | head -1)
mv "$checkpoint" "$OUT/checkpoint.bin"
sha256sum "$OUT/checkpoint.bin" >"$OUT/checkpoint.sha256"

HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
  "$PYTHON" -u "$FEDE/eval_clean.py" \
    --corpus "$ROOT/artifacts/data/corpus_v1.jsonl" \
    --queries "$ROOT/artifacts/data/queries_val_v1.jsonl" \
    --qrels "$ROOT/artifacts/data/qrels_val_v1.tsv" \
    --model BAAI/bge-base-en-v1.5 \
    --checkpoint "$OUT/checkpoint.bin" \
    --pooling cls \
    --device cuda \
    --batch-size 64 \
    --top-k 100 \
    --seed 13 \
    --run-name "b4_fedprox_full_pilot_mu${MU}_val_seed13" \
    --output "$EVAL_OUT"

echo "0" >"$DONE"
echo "[B4-full] completed at $(date -u +%FT%TZ)"
