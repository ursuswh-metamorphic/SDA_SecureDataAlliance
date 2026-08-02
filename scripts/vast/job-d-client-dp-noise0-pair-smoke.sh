#!/bin/bash
set -euo pipefail

ROOT=/workspace/SDA_SecureDataAlliance
FEDE="$ROOT/FedE"
PAIR_ROOT="$ROOT/artifacts/training/d_client_dp_noise0_pair_seed13"
EVAL_ROOT="$ROOT/artifacts/eval/d_client_dp_noise0_pair_seed13"
TASK="$PAIR_ROOT/task"
LOG="$PAIR_ROOT/pair.log"
DONE="$PAIR_ROOT/pair.done"
PYTHON=/workspace/sda-venv/bin/python

mkdir -p "$PAIR_ROOT" "$EVAL_ROOT"
rm -f "$DONE"
exec >>"$LOG" 2>&1

echo "[D-noise0] started at $(date -u +%FT%TZ)"
echo "[D-noise0] paired algorithms=fedavg,client_dp; DP noise=0; clip=off"
echo "[D-noise0] B5 config lr=5e-5, r=8, alpha=16, KD=1; 3x3 smoke"

for METHOD in fedavg client_dp; do
  OUT="$PAIR_ROOT/$METHOD"
  EVAL_OUT="$EVAL_ROOT/${METHOD}_val"
  WORK="$OUT/work"
  mkdir -p "$OUT" "$EVAL_OUT" "$WORK"
  rm -f "$OUT/job.done"

  {
    echo "run_name=d_client_dp_noise0_${METHOD}_seed13"
    echo "algorithm=$METHOD"
    echo "client_dp_mode=noise0"
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

  echo "[D-noise0] method=$METHOD training started at $(date -u +%FT%TZ)"
  cd "$WORK"
  env \
    PYTHONPATH="$FEDE" \
    TRAIN_DATA="$FEDE/selected_data_clean.json" \
    TASK_PATH="$TASK" \
    ALGORITHM="$METHOD" \
    CLIENT_DP_MODE=noise0 \
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
  if [[ -f client_dp_report.json ]]; then
    mv client_dp_report.json "$OUT/client_dp_report.json"
  fi

  echo "[D-noise0] method=$METHOD validation started at $(date -u +%FT%TZ)"
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
      --run-name "d_client_dp_noise0_${METHOD}_val_seed13" \
      --output "$EVAL_OUT"

  echo "0" >"$OUT/job.done"
  echo "[D-noise0] method=$METHOD completed at $(date -u +%FT%TZ)"
done

"$PYTHON" - "$PAIR_ROOT" "$EVAL_ROOT" <<'PY'
import json
import math
import pathlib
import sys

import torch

pair_root = pathlib.Path(sys.argv[1])
eval_root = pathlib.Path(sys.argv[2])
fedavg = torch.load(
    pair_root / "fedavg" / "checkpoint.bin",
    map_location="cpu",
    weights_only=True,
)
noise0 = torch.load(
    pair_root / "client_dp" / "checkpoint.bin",
    map_location="cpu",
    weights_only=True,
)
if set(fedavg) != set(noise0):
    raise RuntimeError("paired checkpoints have different key sets")

max_abs = 0.0
sq_diff = 0.0
sq_ref = 0.0
max_abs_tolerance = 1e-5
relative_l2_tolerance = 1e-5
for key in fedavg:
    left = fedavg[key].double()
    right = noise0[key].double()
    diff = left - right
    max_abs = max(max_abs, diff.abs().max().item())
    sq_diff += diff.pow(2).sum().item()
    sq_ref += left.pow(2).sum().item()
relative_l2 = math.sqrt(sq_diff) / max(math.sqrt(sq_ref), 1e-30)

fedavg_result = json.loads(
    (eval_root / "fedavg_val" / "result.json").read_text(encoding="utf-8")
)
noise0_result = json.loads(
    (eval_root / "client_dp_val" / "result.json").read_text(encoding="utf-8")
)
report = json.loads(
    (pair_root / "client_dp" / "client_dp_report.json").read_text(
        encoding="utf-8"
    )
)
comparison = {
    "gate": "client_dp noise0 must reproduce weighted FedAvg",
    "max_abs_checkpoint_diff": max_abs,
    "relative_l2_checkpoint_diff": relative_l2,
    "max_abs_tolerance": max_abs_tolerance,
    "relative_l2_tolerance": relative_l2_tolerance,
    "fedavg_mrr@10": fedavg_result["aggregate"]["mrr@10"],
    "client_dp_noise0_mrr@10": noise0_result["aggregate"]["mrr@10"],
    "metric_delta": (
        noise0_result["aggregate"]["mrr@10"]
        - fedavg_result["aggregate"]["mrr@10"]
    ),
    "mechanism_releases": report["mechanism_releases"],
    "accountant_steps": report["accountant_steps"],
    "pass": (
        max_abs <= max_abs_tolerance
        and relative_l2 <= relative_l2_tolerance
        and noise0_result["aggregate"] == fedavg_result["aggregate"]
        and report["mechanism_releases"] == 0
        and report["accountant_steps"] == 0
    ),
}
(pair_root / "control_comparison.json").write_text(
    json.dumps(comparison, indent=2) + "\n",
    encoding="utf-8",
)
print(json.dumps(comparison, indent=2))
if not comparison["pass"]:
    raise SystemExit("noise0 control gate failed")
PY

echo "0" >"$DONE"
echo "[D-noise0] completed at $(date -u +%FT%TZ)"
