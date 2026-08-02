#!/bin/bash
set -euo pipefail

ROOT=/workspace/SDA_SecureDataAlliance
FEDE="$ROOT/FedE"
SOURCE="$ROOT/artifacts/data/paper_faithful_source/FEDE4FIN"
TRAIN_DATA="$SOURCE/train_data/data_50000_random.json"
TRAIN_CORPUS="$SOURCE/train_corpus.json"
OUT="$ROOT/artifacts/training/paper_faithful_b5_smoke_seed13"
EVAL_OUT="$ROOT/artifacts/eval/paper_faithful_b5_smoke_val_seed13"
TASK="$OUT/task"
WORK="$OUT/work"
LOG="$OUT/job.log"
DONE="$OUT/job.done"
PYTHON=/workspace/sda-venv/bin/python

mkdir -p "$OUT" "$EVAL_OUT" "$WORK"
rm -f "$DONE"
exec >>"$LOG" 2>&1

echo "[paper-data-smoke] started at $(date -u +%FT%TZ)"

"$PYTHON" "$FEDE/tools/validate_paper_faithful_data.py" \
  --train "$TRAIN_DATA" \
  --train-corpus "$TRAIN_CORPUS" \
  --output "$OUT/data_manifest.json"

{
  echo "run_name=paper_faithful_b5_smoke_seed13"
  echo "data_release=DocAILab/FedE4RAG_Dataset@3983048"
  echo "train_pairs=43658"
  echo "clients=AES,BOEING,ACTIVISIONBLIZZARD,PG,PEPSICO"
  echo "partitioner=paper_five"
  echo "algorithm=fedavg"
  echo "model=BAAI/bge-base-en-v1.5"
  echo "learning_rate=5e-5"
  echo "lora_r=8"
  echo "lora_alpha=16"
  echo "kd_weight=1"
  echo "rounds=3"
  echo "steps_per_round=3"
  echo "batch_size=16"
  echo "seed=13"
  sha256sum \
    "$TRAIN_DATA" \
    "$TRAIN_CORPUS" \
    "$FEDE/main_lora.py" \
    "$FEDE/flgo/benchmark/partition.py" \
    "$FEDE/flgo/benchmark/fedrag_classification/core.py" \
    "$FEDE/flgo/algorithm/fedrag_lora.py"
} >"$OUT/manifest.txt"

cd "$WORK"
env \
  PYTHONPATH="$FEDE" \
  TRAIN_DATA="$TRAIN_DATA" \
  TASK_PATH="$TASK" \
  PARTITIONER=paper_five \
  ALGORITHM=fedavg \
  NUM_CLIENTS=5 \
  USE_LORA=1 \
  USE_QLORA=0 \
  LORA_R=8 \
  LORA_ALPHA=16 \
  LORA_DROPOUT=0.05 \
  LEARNING_RATE=5e-5 \
  DP_ENABLED=0 \
  KD_WEIGHT=1 \
  POOLING=cls \
  NUM_ROUNDS=3 \
  NUM_STEPS=3 \
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
    --lora-checkpoint "$OUT/checkpoint.bin" \
    --lora-r 8 \
    --lora-alpha 16 \
    --lora-dropout 0.05 \
    --pooling cls \
    --device cuda \
    --batch-size 64 \
    --top-k 100 \
    --seed 13 \
    --run-name paper_faithful_b5_smoke_val_seed13 \
    --output "$EVAL_OUT"

echo "0" >"$DONE"
echo "[paper-data-smoke] completed at $(date -u +%FT%TZ)"
