#!/bin/bash
set -euo pipefail

ROOT=/workspace/SDA_SecureDataAlliance
FEDE="$ROOT/FedE"
KD_SUMMARY="$ROOT/artifacts/training/b5_lora_kd_sweep_seed13/sweep_summary.json"
TRAIN_ROOT="$ROOT/artifacts/training/b5_lora_full_pair_seed13"
EVAL_ROOT="$ROOT/artifacts/eval/b5_lora_full_pair_seed13"
TASK="$TRAIN_ROOT/task"
LOG="$TRAIN_ROOT/pair.log"
DONE="$TRAIN_ROOT/pair.done"
KDS=(0 1.0)
PYTHON=/workspace/sda-venv/bin/python

read -r LR RANK ALPHA < <("$PYTHON" -c \
  "import json; s=json.load(open('$KD_SUMMARY', encoding='utf-8')); print(s['best_learning_rate'], s['best_lora_r'], s['best_lora_alpha'])")

mkdir -p "$TRAIN_ROOT" "$EVAL_ROOT"
rm -f "$DONE"
exec >>"$LOG" 2>&1

echo "[B5-full-pair] started at $(date -u +%FT%TZ)"
echo "[B5-full-pair] validation-only; DP=0; lr=$LR; rank=$RANK; alpha=$ALPHA"
echo "[B5-full-pair] kd_weights=${KDS[*]}, rounds=15, steps=50, batch=16, seed=13"

for KD in "${KDS[@]}"; do
  TAG="${KD//./p}"
  OUT="$TRAIN_ROOT/kw${TAG}"
  EVAL_OUT="$EVAL_ROOT/kw${TAG}_val"
  WORK="$OUT/work"

  mkdir -p "$OUT" "$EVAL_OUT" "$WORK"
  rm -f "$OUT/job.done"

  {
    echo "run_name=b5_lora_full_kw${KD}_seed13"
    echo "protocol=5 company-exclusive clients; weighted FedAvg; LoRA+KD"
    echo "learning_rate=$LR"
    echo "lora_r=$RANK"
    echo "lora_alpha=$ALPHA"
    echo "lora_dropout=0.05"
    echo "kd_weight=$KD"
    echo "dp_enabled=0"
    echo "rounds=15"
    echo "steps_per_round=50"
    echo "batch_size=16"
    echo "seed=13"
    sha256sum \
      "$FEDE/main_lora.py" \
      "$FEDE/lora_hparams.py" \
      "$FEDE/eval_clean.py" \
      "$FEDE/flgo/algorithm/fedrag_lora.py" \
      "$FEDE/flgo/benchmark/fedrag_classification/core.py" \
      "$FEDE/flgo/benchmark/fedrag_classification/config.py" \
      "$FEDE/selected_data_clean.json" \
      "$KD_SUMMARY"
  } >"$OUT/manifest.txt"

  echo "[B5-full-pair] kw=$KD training started at $(date -u +%FT%TZ)"
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

  echo "[B5-full-pair] kw=$KD validation started at $(date -u +%FT%TZ)"
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
      --run-name "b5_lora_full_kw${KD}_val_seed13" \
      --output "$EVAL_OUT"

  echo "0" >"$OUT/job.done"
  echo "[B5-full-pair] kw=$KD completed at $(date -u +%FT%TZ)"
done

"$PYTHON" - "$EVAL_ROOT" "$TRAIN_ROOT/pair_summary.json" \
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
for kd, tag in ((0.0, "0"), (1.0, "1p0")):
    result_path = eval_root / f"kw{tag}_val" / "result.json"
    result = json.loads(result_path.read_text(encoding="utf-8"))
    metrics = result["aggregate"]
    rows.append({
        "kd_weight": kd,
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
    "learning_rate": learning_rate,
    "lora_r": rank,
    "lora_alpha": alpha,
    "best_kd_weight": rows[0]["kd_weight"],
    "results_ranked": rows,
}
summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
print(json.dumps(summary, indent=2))
PY

echo "0" >"$DONE"
echo "[B5-full-pair] completed at $(date -u +%FT%TZ)"
