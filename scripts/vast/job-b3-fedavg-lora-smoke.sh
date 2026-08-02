#!/bin/bash
set -euo pipefail

ROOT=/workspace/SDA_SecureDataAlliance
FEDE="$ROOT/FedE"
OUT="$ROOT/artifacts/training/b3_fedavg_lora_smoke_seed13"
EVAL_OUT="$ROOT/artifacts/eval/b3_fedavg_lora_smoke_val_seed13"
WORK="$OUT/work"
LOG="$OUT/job.log"
DONE="$OUT/job.done"

mkdir -p "$OUT" "$EVAL_OUT" "$WORK"
rm -f "$DONE"
cd "$WORK"

{
  echo "run_name=b3_fedavg_lora_smoke_seed13"
  echo "protocol=5 company-exclusive clients; datavol-weighted FedAvg; LoRA"
  echo "model=BAAI/bge-base-en-v1.5"
  echo "pooling=cls"
  echo "num_clients=5"
  echo "use_lora=1"
  echo "dp_enabled=0"
  echo "kd_weight=0"
  echo "rounds=3"
  echo "steps_per_round=3"
  echo "batch_size=16"
  echo "learning_rate=1e-5"
  echo "seed=13"
  sha256sum \
    "$FEDE/main_lora.py" \
    "$FEDE/eval_clean.py" \
    "$FEDE/flgo/algorithm/fedrag_lora.py" \
    "$FEDE/flgo/benchmark/fedrag_classification/config.py" \
    "$FEDE/selected_data_clean.json"
} >"$OUT/manifest.txt"

exec >>"$LOG" 2>&1
echo "[B3-LoRA] started at $(date -u +%FT%TZ)"

env \
  PYTHONPATH="$FEDE" \
  TRAIN_DATA="$FEDE/selected_data_clean.json" \
  TASK_PATH="$OUT/task" \
  NUM_CLIENTS=5 \
  USE_LORA=1 \
  USE_QLORA=0 \
  DP_ENABLED=0 \
  KD_WEIGHT=0 \
  POOLING=cls \
  NUM_ROUNDS=3 \
  NUM_STEPS=3 \
  BATCH_SIZE=16 \
  GPU=0 \
  SEED=13 \
  DATA_SEED=13 \
  PYTHONUNBUFFERED=1 \
  /workspace/sda-venv/bin/python -u "$FEDE/main_lora.py"

checkpoint=$(ls -1t x-lora_*.bin | head -1)
mv "$checkpoint" "$OUT/checkpoint.bin"
sha256sum "$OUT/checkpoint.bin" >"$OUT/checkpoint.sha256"

/workspace/sda-venv/bin/python -u "$FEDE/eval_clean.py" \
  --corpus "$ROOT/artifacts/data/corpus_v1.jsonl" \
  --queries "$ROOT/artifacts/data/queries_val_v1.jsonl" \
  --qrels "$ROOT/artifacts/data/qrels_val_v1.tsv" \
  --model BAAI/bge-base-en-v1.5 \
  --lora-checkpoint "$OUT/checkpoint.bin" \
  --pooling cls \
  --device cuda \
  --batch-size 64 \
  --top-k 100 \
  --seed 13 \
  --run-name b3_fedavg_lora_smoke_val_seed13 \
  --output "$EVAL_OUT"

echo "0" >"$DONE"
echo "[B3-LoRA] completed at $(date -u +%FT%TZ)"
