#!/bin/bash
set -euo pipefail

ROOT=/workspace/SDA_SecureDataAlliance
FEDE="$ROOT/FedE"
SWEEP_ROOT="$ROOT/artifacts/training/d_client_dp_epsilon_smoke_seed13"
EVAL_ROOT="$ROOT/artifacts/eval/d_client_dp_epsilon_smoke_seed13"
TASK="$SWEEP_ROOT/task"
LOG="$SWEEP_ROOT/sweep.log"
DONE="$SWEEP_ROOT/sweep.done"
PYTHON=/workspace/sda-venv/bin/python

mkdir -p "$SWEEP_ROOT" "$EVAL_ROOT"
rm -f "$DONE"
exec >>"$LOG" 2>&1

echo "[D-epsilon-smoke] started at $(date -u +%FT%TZ)"
echo "[D-epsilon-smoke] epsilons=20,8,3,1; C=8; delta=1e-5; q=1; 3x3"

for EPSILON in 20 8 3 1; do
  TAG="eps${EPSILON}"
  OUT="$SWEEP_ROOT/$TAG"
  EVAL_OUT="$EVAL_ROOT/$TAG"
  WORK="$OUT/work"
  mkdir -p "$OUT" "$EVAL_OUT" "$WORK"
  if [[ -f "$OUT/job.done" && -f "$EVAL_OUT/result.json" ]]; then
    echo "[D-epsilon-smoke] epsilon=$EPSILON already complete; skipping"
    continue
  fi

  {
    echo "run_name=d_client_dp_${TAG}_smoke_seed13"
    echo "algorithm=client_dp"
    echo "client_dp_mode=dp"
    echo "privacy_unit=one_complete_federated_client"
    echo "adjacency=replace_one"
    echo "client_dp_clip_norm=8"
    echo "client_dp_target_epsilon=$EPSILON"
    echo "client_dp_target_delta=1e-5"
    echo "clients_per_round=5"
    echo "num_clients=5"
    echo "sampling_rate=1"
    echo "mechanism_releases=3"
    echo "noise_rng=os_entropy"
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
      "$FEDE/privacy/rdp_accountant.py" \
      "$FEDE/flgo/algorithm/fedrag_client_dp.py" \
      "$FEDE/flgo/algorithm/fedrag_lora.py" \
      "$FEDE/selected_data_clean.json"
  } >"$OUT/manifest.txt"

  if [[ -f "$OUT/checkpoint.bin" && -f "$OUT/client_dp_report.json" ]]; then
    echo "[D-epsilon-smoke] epsilon=$EPSILON checkpoint exists; skipping training"
  else
    echo "[D-epsilon-smoke] epsilon=$EPSILON training started at $(date -u +%FT%TZ)"
    cd "$WORK"
    env \
      PYTHONPATH="$FEDE" \
      TRAIN_DATA="$FEDE/selected_data_clean.json" \
      TASK_PATH="$TASK" \
      ALGORITHM=client_dp \
      CLIENT_DP_MODE=dp \
      CLIENT_DP_CLIP_NORM=8 \
      CLIENT_DP_TARGET_EPS="$EPSILON" \
      CLIENT_DP_TARGET_DELTA=1e-5 \
      CLIENT_DP_CLIENTS_PER_ROUND=5 \
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
  fi

  if [[ -f "$EVAL_OUT/result.json" ]]; then
    echo "[D-epsilon-smoke] epsilon=$EPSILON validation exists; skipping"
  else
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
        --run-name "d_client_dp_${TAG}_smoke_val_seed13" \
        --output "$EVAL_OUT"
  fi

  echo "0" >"$OUT/job.done"
  echo "[D-epsilon-smoke] epsilon=$EPSILON completed at $(date -u +%FT%TZ)"
done

"$PYTHON" - "$SWEEP_ROOT" "$EVAL_ROOT" <<'PY'
import json
import pathlib
import sys

training_root = pathlib.Path(sys.argv[1])
eval_root = pathlib.Path(sys.argv[2])
rows = []
for epsilon in (20, 8, 3, 1):
    tag = f"eps{epsilon}"
    report = json.loads(
        (training_root / tag / "client_dp_report.json").read_text()
    )
    result = json.loads((eval_root / tag / "result.json").read_text())
    final_round = report["rounds"][-1]
    rows.append(
        {
            "target_epsilon": epsilon,
            "final_epsilon": report["final_epsilon"],
            "delta": report["target_delta"],
            "noise_multiplier": final_round["noise_multiplier"],
            "sensitivity": final_round["sensitivity"],
            "noise_std": final_round["noise_std"],
            "mechanism_releases": report["mechanism_releases"],
            "accountant_steps": report["accountant_steps"],
            "mrr@10": result["aggregate"]["mrr@10"],
            "hit@1": result["aggregate"]["hit@1"],
            "hit@10": result["aggregate"]["hit@10"],
            "ndcg@10": result["aggregate"]["ndcg@10"],
        }
    )
summary = {
    "privacy_unit": "one complete federated client",
    "adjacency": "replace-one",
    "clip_norm": 8.0,
    "sample_rate": 1.0,
    "rounds": 3,
    "steps_per_round": 3,
    "delta": 1e-5,
    "noise0_mrr@10_reference": 25.52,
    "clip_only_mrr@10_reference": 25.52,
    "rows": rows,
}
(training_root / "sweep_summary.json").write_text(
    json.dumps(summary, indent=2) + "\n",
    encoding="utf-8",
)
print(json.dumps(summary, indent=2))
PY

echo "0" >"$DONE"
echo "[D-epsilon-smoke] completed at $(date -u +%FT%TZ)"
