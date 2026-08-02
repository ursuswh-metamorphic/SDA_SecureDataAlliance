#!/bin/bash
set -euo pipefail

ROOT=/workspace/SDA_SecureDataAlliance
FEDE="$ROOT/FedE"
OUT="$ROOT/artifacts/training/d_client_dp_clip_only_c8_seed13"
EVAL_OUT="$ROOT/artifacts/eval/d_client_dp_clip_only_c8_seed13/val"
TASK="$OUT/task"
WORK="$OUT/work"
LOG="$OUT/pilot.log"
DONE="$OUT/pilot.done"
PYTHON=/workspace/sda-venv/bin/python

mkdir -p "$OUT" "$EVAL_OUT" "$WORK"
rm -f "$DONE" "$OUT/job.done"
exec >>"$LOG" 2>&1

echo "[D-clip-only] started at $(date -u +%FT%TZ)"
echo "[D-clip-only] C=8; B5 config lr=5e-5, r=8, alpha=16, KD=1; 3x3 pilot"

{
  echo "run_name=d_client_dp_clip_only_c8_seed13"
  echo "algorithm=client_dp"
  echo "client_dp_mode=clip_only"
  echo "client_dp_clip_norm=8"
  echo "learning_rate=5e-5"
  echo "lora_r=8"
  echo "lora_alpha=16"
  echo "kd_weight=1"
  echo "record_dp_enabled=0"
  echo "rounds=3"
  echo "steps_per_round=3"
  echo "batch_size=16"
  echo "seed=13"
  sha256sum \
    "$FEDE/main_lora.py" \
    "$FEDE/client_dp_utils.py" \
    "$FEDE/flgo/algorithm/fedrag_client_dp.py" \
    "$FEDE/flgo/algorithm/fedrag_lora.py" \
    "$FEDE/selected_data_clean.json"
} >"$OUT/manifest.txt"

cd "$WORK"
env \
  PYTHONPATH="$FEDE" \
  TRAIN_DATA="$FEDE/selected_data_clean.json" \
  TASK_PATH="$TASK" \
  ALGORITHM=client_dp \
  CLIENT_DP_MODE=clip_only \
  CLIENT_DP_CLIP_NORM=8 \
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
mv client_dp_report.json "$OUT/client_dp_report.json"
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
    --run-name d_client_dp_clip_only_c8_val_seed13 \
    --output "$EVAL_OUT"

"$PYTHON" - "$OUT" "$EVAL_OUT" <<'PY'
import json
import pathlib
import sys

out = pathlib.Path(sys.argv[1])
eval_out = pathlib.Path(sys.argv[2])
report = json.loads((out / "client_dp_report.json").read_text())
result = json.loads((eval_out / "result.json").read_text())
coefficients = [
    coefficient
    for round_report in report["rounds"]
    for coefficient in round_report["clip_coefficients"]
]
summary = {
    "mode": report["mode"],
    "clip_norm": 8.0,
    "formal_dp_claim": report["formal_dp_claim"],
    "mechanism_releases": report["mechanism_releases"],
    "accountant_steps": report["accountant_steps"],
    "total_client_updates": len(coefficients),
    "clipped_client_updates": sum(value < 1.0 for value in coefficients),
    "min_clip_coefficient": min(coefficients),
    "mrr@10": result["aggregate"]["mrr@10"],
    "noise0_mrr@10_reference": 25.52,
    "mrr@10_delta_vs_noise0": result["aggregate"]["mrr@10"] - 25.52,
}
(out / "pilot_summary.json").write_text(
    json.dumps(summary, indent=2) + "\n",
    encoding="utf-8",
)
print(json.dumps(summary, indent=2))
PY

echo "0" >"$OUT/job.done"
echo "0" >"$DONE"
echo "[D-clip-only] completed at $(date -u +%FT%TZ)"
