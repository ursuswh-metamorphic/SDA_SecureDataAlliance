#!/bin/bash
set -euo pipefail

ROOT=/workspace/SDA_SecureDataAlliance
FEDE="$ROOT/FedE"
SWEEP_ROOT="$ROOT/artifacts/training/b5_lora_lr_sweep_seed13"
EVAL_ROOT="$ROOT/artifacts/eval/b5_lora_lr_sweep_seed13"
TASK="$SWEEP_ROOT/task"
LOG="$SWEEP_ROOT/sweep.log"
DONE="$SWEEP_ROOT/sweep.done"
LRS=(5e-5 1e-4 2e-4)

mkdir -p "$SWEEP_ROOT" "$EVAL_ROOT"
rm -f "$DONE"
exec >>"$LOG" 2>&1

echo "[B5-LR] started at $(date -u +%FT%TZ)"
echo "[B5-LR] validation-only; DP=0; KD=0; rank=8; alpha=16"
echo "[B5-LR] learning_rates=${LRS[*]}, rounds=5, steps=20, batch=16, seed=13"

for LR in "${LRS[@]}"; do
  TAG="${LR//./p}"
  TAG="${TAG//-/m}"
  OUT="$SWEEP_ROOT/lr${TAG}"
  EVAL_OUT="$EVAL_ROOT/lr${TAG}_val"
  WORK="$OUT/work"

  mkdir -p "$OUT" "$EVAL_OUT" "$WORK"
  rm -f "$OUT/job.done"

  {
    echo "run_name=b5_lora_lr_sweep_lr${LR}_seed13"
    echo "protocol=5 company-exclusive clients; weighted FedAvg; LoRA"
    echo "learning_rate=$LR"
    echo "lora_r=8"
    echo "lora_alpha=16"
    echo "lora_dropout=0.05"
    echo "kd_weight=0"
    echo "dp_enabled=0"
    echo "rounds=5"
    echo "steps_per_round=20"
    echo "batch_size=16"
    echo "seed=13"
    sha256sum \
      "$FEDE/main_lora.py" \
      "$FEDE/lora_hparams.py" \
      "$FEDE/flgo/algorithm/fedrag_lora.py" \
      "$FEDE/flgo/benchmark/fedrag_classification/config.py" \
      "$FEDE/selected_data_clean.json"
  } >"$OUT/manifest.txt"

  echo "[B5-LR] lr=$LR training started at $(date -u +%FT%TZ)"
  cd "$WORK"
  env \
    PYTHONPATH="$FEDE" \
    TRAIN_DATA="$FEDE/selected_data_clean.json" \
    TASK_PATH="$TASK" \
    ALGORITHM=fedavg \
    NUM_CLIENTS=5 \
    USE_LORA=1 \
    USE_QLORA=0 \
    LORA_R=8 \
    LORA_ALPHA=16 \
    LORA_DROPOUT=0.05 \
    LEARNING_RATE="$LR" \
    DP_ENABLED=0 \
    KD_WEIGHT=0 \
    POOLING=cls \
    NUM_ROUNDS=5 \
    NUM_STEPS=20 \
    BATCH_SIZE=16 \
    GPU=0 \
    SEED=13 \
    DATA_SEED=13 \
    HF_HUB_OFFLINE=1 \
    TRANSFORMERS_OFFLINE=1 \
    PYTHONUNBUFFERED=1 \
    /workspace/sda-venv/bin/python -u "$FEDE/main_lora.py"

  checkpoint=$(ls -1t x-lora_*.bin | head -1)
  mv "$checkpoint" "$OUT/checkpoint.bin"
  sha256sum "$OUT/checkpoint.bin" >"$OUT/checkpoint.sha256"

  echo "[B5-LR] lr=$LR validation started at $(date -u +%FT%TZ)"
  HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
    /workspace/sda-venv/bin/python -u "$FEDE/eval_clean.py" \
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
      --run-name "b5_lora_lr_sweep_lr${LR}_val_seed13" \
      --output "$EVAL_OUT"

  echo "0" >"$OUT/job.done"
  echo "[B5-LR] lr=$LR completed at $(date -u +%FT%TZ)"
done

/workspace/sda-venv/bin/python - "$EVAL_ROOT" "$SWEEP_ROOT/sweep_summary.json" <<'PY'
import json
import pathlib
import sys

eval_root = pathlib.Path(sys.argv[1])
summary_path = pathlib.Path(sys.argv[2])
rows = []
for lr in ("5e-5", "1e-4", "2e-4"):
    tag = lr.replace(".", "p").replace("-", "m")
    result_path = eval_root / f"lr{tag}_val" / "result.json"
    result = json.loads(result_path.read_text(encoding="utf-8"))
    metrics = result["aggregate"]
    rows.append({
        "learning_rate": float(lr),
        "mrr@10": metrics["mrr@10"],
        "hit@1": metrics["hit@1"],
        "hit@10": metrics["hit@10"],
        "ndcg@10": metrics["ndcg@10"],
        "result": str(result_path),
    })
rows.sort(key=lambda row: (-row["mrr@10"], -row["ndcg@10"], row["learning_rate"]))
summary = {
    "selection_metric": "mrr@10",
    "validation_only": True,
    "test_split_locked": True,
    "dp_enabled": False,
    "kd_weight": 0.0,
    "lora_r": 8,
    "lora_alpha": 16,
    "best_learning_rate": rows[0]["learning_rate"],
    "results_ranked": rows,
}
summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
print(json.dumps(summary, indent=2))
PY

echo "0" >"$DONE"
echo "[B5-LR] completed at $(date -u +%FT%TZ)"
