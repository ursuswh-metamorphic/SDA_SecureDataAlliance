#!/bin/bash
set -euo pipefail

ROOT=/workspace/SDA_SecureDataAlliance
FEDE="$ROOT/FedE"
RANK_SUMMARY="$ROOT/artifacts/training/b5_lora_rank_sweep_seed13/sweep_summary.json"
RANK_ROOT="$ROOT/artifacts/training/b5_lora_rank_sweep_seed13"
SWEEP_ROOT="$ROOT/artifacts/training/b5_lora_kd_sweep_seed13"
EVAL_ROOT="$ROOT/artifacts/eval/b5_lora_kd_sweep_seed13"
TASK="$SWEEP_ROOT/task"
LOG="$SWEEP_ROOT/sweep.log"
DONE="$SWEEP_ROOT/sweep.done"
KDS=(0 0.1 0.5 1.0)
PYTHON=/workspace/sda-venv/bin/python

read -r LR RANK ALPHA < <("$PYTHON" -c \
  "import json; s=json.load(open('$RANK_SUMMARY', encoding='utf-8')); print(s['best_learning_rate'], s['best_lora_r'], s['best_lora_alpha'])")

mkdir -p "$SWEEP_ROOT" "$EVAL_ROOT"
rm -f "$DONE"
exec >>"$LOG" 2>&1

echo "[B5-KD] started at $(date -u +%FT%TZ)"
echo "[B5-KD] validation-only; DP=0; lr=$LR; rank=$RANK; alpha=$ALPHA"
echo "[B5-KD] kd_weights=${KDS[*]}, rounds=5, steps=20, batch=16, seed=13"

for KD in "${KDS[@]}"; do
  TAG="${KD//./p}"
  OUT="$SWEEP_ROOT/kw${TAG}"
  EVAL_OUT="$EVAL_ROOT/kw${TAG}_val"
  WORK="$OUT/work"

  mkdir -p "$OUT" "$EVAL_OUT" "$WORK"
  rm -f "$OUT/job.done"

  {
    echo "run_name=b5_lora_kd_sweep_kw${KD}_seed13"
    echo "protocol=5 company-exclusive clients; weighted FedAvg; LoRA+KD"
    echo "learning_rate=$LR"
    echo "lora_r=$RANK"
    echo "lora_alpha=$ALPHA"
    echo "lora_dropout=0.05"
    echo "kd_weight=$KD"
    echo "dp_enabled=0"
    echo "rounds=5"
    echo "steps_per_round=20"
    echo "batch_size=16"
    echo "seed=13"
    sha256sum \
      "$FEDE/main_lora.py" \
      "$FEDE/lora_hparams.py" \
      "$FEDE/flgo/algorithm/fedrag_lora.py" \
      "$FEDE/flgo/benchmark/fedrag_classification/core.py" \
      "$FEDE/flgo/benchmark/fedrag_classification/config.py" \
      "$FEDE/selected_data_clean.json" \
      "$RANK_SUMMARY"
  } >"$OUT/manifest.txt"

  if [[ "$KD" == "0" ]]; then
    echo "[B5-KD] kw=0 reusing selected no-KD checkpoint"
    cp "$RANK_ROOT/r${RANK}/checkpoint.bin" "$OUT/checkpoint.bin"
    sha256sum "$OUT/checkpoint.bin" >"$OUT/checkpoint.sha256"
  else
    echo "[B5-KD] kw=$KD training started at $(date -u +%FT%TZ)"
    cd "$WORK"
    env \
      PYTHONPATH="$FEDE" \
      TRAIN_DATA="$FEDE/selected_data_clean.json" \
      TASK_PATH="$TASK" \
      ALGORITHM=fedavg \
      NUM_CLIENTS=5 \
      USE_LORA=1 \
      USE_QLORA=0 \
      LORA_R="$RANK" \
      LORA_ALPHA="$ALPHA" \
      LORA_DROPOUT=0.05 \
      LEARNING_RATE="$LR" \
      DP_ENABLED=0 \
      KD_WEIGHT="$KD" \
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
      "$PYTHON" -u "$FEDE/main_lora.py"

    checkpoint=$(ls -1t x-lora_*.bin | head -1)
    mv "$checkpoint" "$OUT/checkpoint.bin"
    sha256sum "$OUT/checkpoint.bin" >"$OUT/checkpoint.sha256"
  fi

  echo "[B5-KD] kw=$KD validation started at $(date -u +%FT%TZ)"
  HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
    "$PYTHON" -u "$FEDE/eval_clean.py" \
      --corpus "$ROOT/artifacts/data/corpus_v1.jsonl" \
      --queries "$ROOT/artifacts/data/queries_val_v1.jsonl" \
      --qrels "$ROOT/artifacts/data/qrels_val_v1.tsv" \
      --model BAAI/bge-base-en-v1.5 \
      --lora-checkpoint "$OUT/checkpoint.bin" \
      --lora-r "$RANK" \
      --lora-alpha "$ALPHA" \
      --lora-dropout 0.05 \
      --pooling cls \
      --device cuda \
      --batch-size 64 \
      --top-k 100 \
      --seed 13 \
      --run-name "b5_lora_kd_sweep_kw${KD}_val_seed13" \
      --output "$EVAL_OUT"

  echo "0" >"$OUT/job.done"
  echo "[B5-KD] kw=$KD completed at $(date -u +%FT%TZ)"
done

"$PYTHON" - "$EVAL_ROOT" "$SWEEP_ROOT/sweep_summary.json" \
  "$LR" "$RANK" "$ALPHA" <<'PY'
import json
import pathlib
import sys

eval_root = pathlib.Path(sys.argv[1])
summary_path = pathlib.Path(sys.argv[2])
learning_rate = float(sys.argv[3])
rank = int(sys.argv[4])
alpha = int(sys.argv[5])
rows = []
for kd in (0.0, 0.1, 0.5, 1.0):
    tag = str(kd).replace(".", "p")
    if kd == 0.0:
        tag = "0"
    result_path = eval_root / f"kw{tag}_val" / "result.json"
    result = json.loads(result_path.read_text(encoding="utf-8"))
    metrics = result["aggregate"]
    rows.append({
        "kd_weight": kd,
        "lora_r": rank,
        "lora_alpha": alpha,
        "learning_rate": learning_rate,
        "mrr@10": metrics["mrr@10"],
        "hit@1": metrics["hit@1"],
        "hit@10": metrics["hit@10"],
        "ndcg@10": metrics["ndcg@10"],
        "result": str(result_path),
    })
rows.sort(key=lambda row: (-row["mrr@10"], -row["ndcg@10"], row["kd_weight"]))
summary = {
    "selection_metric": "mrr@10",
    "validation_only": True,
    "test_split_locked": True,
    "dp_enabled": False,
    "best_learning_rate": learning_rate,
    "best_lora_r": rank,
    "best_lora_alpha": alpha,
    "best_kd_weight": rows[0]["kd_weight"],
    "results_ranked": rows,
}
summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
print(json.dumps(summary, indent=2))
PY

echo "0" >"$DONE"
echo "[B5-KD] completed at $(date -u +%FT%TZ)"
