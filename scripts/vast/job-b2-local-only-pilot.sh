#!/bin/bash
set -euo pipefail

ROOT=/workspace/SDA_SecureDataAlliance
FEDE="$ROOT/FedE"
SPLITS="$ROOT/artifacts/data/b2_local_splits"
TRAIN_ROOT="$ROOT/artifacts/training/b2_local_only_seed13"
EVAL_ROOT="$ROOT/artifacts/eval/b2_local_only_val_seed13"
MERGED="$ROOT/artifacts/eval/b2_local_only_company_routed_val_seed13"
LOG="$TRAIN_ROOT/job.log"
DONE="$TRAIN_ROOT/job.done"

mkdir -p "$TRAIN_ROOT" "$EVAL_ROOT" "$MERGED"
rm -f "$DONE"

{
  echo "run_name=b2_local_only_company_routed_seed13"
  echo "protocol=5 independent full-finetune clients; company-routed validation"
  echo "model=BAAI/bge-base-en-v1.5"
  echo "pooling=cls"
  echo "use_lora=0"
  echo "dp_enabled=0"
  echo "kd_weight=0"
  echo "rounds_per_client=15"
  echo "steps_per_round=50"
  echo "batch_size=16"
  echo "learning_rate=1e-5"
  echo "seed=13"
  sha256sum \
    "$FEDE/main_lora.py" \
    "$FEDE/flgo/algorithm/fedrag_lora.py" \
    "$FEDE/flgo/benchmark/fedrag_classification/core.py" \
    "$FEDE/tools/build_local_only_splits.py" \
    "$FEDE/tools/merge_routed_eval.py" \
    "$SPLITS/manifest.json"
} >"$TRAIN_ROOT/manifest.txt"

exec >>"$LOG" 2>&1
echo "[B2] started at $(date -u +%FT%TZ)"

for client_id in 0 1 2 3 4; do
  client_out="$TRAIN_ROOT/client_$client_id"
  work="$client_out/work"
  mkdir -p "$work"
  cd "$work"
  echo "[B2] training client $client_id at $(date -u +%FT%TZ)"

  env \
    PYTHONPATH="$FEDE" \
    TRAIN_DATA="$SPLITS/client_${client_id}_train.json" \
    TASK_PATH="$client_out/task" \
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
    /workspace/sda-venv/bin/python -u "$FEDE/main_lora.py"

  checkpoint=$(ls -1t x-lora_*.bin | head -1)
  mv "$checkpoint" "$client_out/checkpoint.bin"
  sha256sum "$client_out/checkpoint.bin" >"$client_out/checkpoint.sha256"
  echo "[B2] client $client_id trained at $(date -u +%FT%TZ)"
done

for client_id in 0 1 2 3 4; do
  eval_out="$EVAL_ROOT/client_$client_id"
  mkdir -p "$eval_out"
  echo "[B2] evaluating client $client_id at $(date -u +%FT%TZ)"
  /workspace/sda-venv/bin/python -u "$FEDE/eval_clean.py" \
    --corpus "$ROOT/artifacts/data/corpus_v1.jsonl" \
    --queries "$SPLITS/client_${client_id}_queries.jsonl" \
    --qrels "$ROOT/artifacts/data/qrels_val_v1.tsv" \
    --model BAAI/bge-base-en-v1.5 \
    --checkpoint "$TRAIN_ROOT/client_$client_id/checkpoint.bin" \
    --pooling cls \
    --device cuda \
    --batch-size 64 \
    --top-k 100 \
    --seed 13 \
    --run-name "b2_local_client_${client_id}_val_seed13" \
    --output "$eval_out"
done

/workspace/sda-venv/bin/python -u "$FEDE/tools/merge_routed_eval.py" \
  --eval-dir "$EVAL_ROOT/client_0" \
  --eval-dir "$EVAL_ROOT/client_1" \
  --eval-dir "$EVAL_ROOT/client_2" \
  --eval-dir "$EVAL_ROOT/client_3" \
  --eval-dir "$EVAL_ROOT/client_4" \
  --queries "$ROOT/artifacts/data/queries_val_v1.jsonl" \
  --output "$MERGED" \
  --run-name b2_local_only_company_routed_val_seed13

echo "0" >"$DONE"
echo "[B2] completed at $(date -u +%FT%TZ)"
