#!/bin/bash
set -euo pipefail

ROOT=/workspace/SDA_SecureDataAlliance
FEDE="$ROOT/FedE"
SWEEP_ROOT="$ROOT/artifacts/training/b4_fedprox_mu_sweep_seed13"
EVAL_ROOT="$ROOT/artifacts/eval/b4_fedprox_mu_sweep_seed13"
TASK="$SWEEP_ROOT/task"
SWEEP_LOG="$SWEEP_ROOT/sweep.log"
SWEEP_DONE="$SWEEP_ROOT/sweep.done"
MUS=(0.001 0.01 0.1)

mkdir -p "$SWEEP_ROOT" "$EVAL_ROOT"
rm -f "$SWEEP_DONE"
exec >>"$SWEEP_LOG" 2>&1

echo "[B4-sweep] started at $(date -u +%FT%TZ)"
echo "[B4-sweep] protocol=5 clients; full fine-tune; validation only"
echo "[B4-sweep] mus=${MUS[*]}, rounds=5, steps=20, batch=16, seed=13"

for MU in "${MUS[@]}"; do
  TAG="${MU//./p}"
  OUT="$SWEEP_ROOT/mu${TAG}"
  EVAL_OUT="$EVAL_ROOT/mu${TAG}_val"
  WORK="$OUT/work"
  DONE="$OUT/job.done"

  mkdir -p "$OUT" "$EVAL_OUT" "$WORK"
  rm -f "$DONE"

  {
    echo "run_name=b4_fedprox_mu_sweep_mu${MU}_seed13"
    echo "protocol=5 company-exclusive clients; FedProx; full fine-tune"
    echo "fedprox_mu=$MU"
    echo "model=BAAI/bge-base-en-v1.5"
    echo "pooling=cls"
    echo "rounds=5"
    echo "steps_per_round=20"
    echo "batch_size=16"
    echo "learning_rate=1e-5"
    echo "seed=13"
    sha256sum \
      "$FEDE/main_lora.py" \
      "$FEDE/flgo/algorithm/fedrag_fedprox.py" \
      "$FEDE/flgo/algorithm/fedprox_utils.py" \
      "$FEDE/selected_data_clean.json"
  } >"$OUT/manifest.txt"

  echo "[B4-sweep] mu=$MU training started at $(date -u +%FT%TZ)"
  cd "$WORK"
  env \
    PYTHONPATH="$FEDE" \
    TRAIN_DATA="$FEDE/selected_data_clean.json" \
    TASK_PATH="$TASK" \
    ALGORITHM=fedprox \
    FEDPROX_MU="$MU" \
    NUM_CLIENTS=5 \
    USE_LORA=0 \
    USE_QLORA=0 \
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

  echo "[B4-sweep] mu=$MU validation started at $(date -u +%FT%TZ)"
  HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
    /workspace/sda-venv/bin/python -u "$FEDE/eval_clean.py" \
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
      --run-name "b4_fedprox_mu_sweep_mu${MU}_val_seed13" \
      --output "$EVAL_OUT"

  echo "0" >"$DONE"
  echo "[B4-sweep] mu=$MU completed at $(date -u +%FT%TZ)"
done

/workspace/sda-venv/bin/python - "$EVAL_ROOT" "$SWEEP_ROOT/sweep_summary.json" <<'PY'
import json
import pathlib
import sys

eval_root = pathlib.Path(sys.argv[1])
summary_path = pathlib.Path(sys.argv[2])
rows = []
for mu in ("0.001", "0.01", "0.1"):
    tag = mu.replace(".", "p")
    result_path = eval_root / f"mu{tag}_val" / "result.json"
    result = json.loads(result_path.read_text(encoding="utf-8"))
    aggregate = result["aggregate"]
    rows.append({
        "mu": float(mu),
        "mrr@10": aggregate["mrr@10"],
        "hit@1": aggregate["hit@1"],
        "hit@10": aggregate["hit@10"],
        "ndcg@10": aggregate["ndcg@10"],
        "result": str(result_path),
    })
rows.sort(key=lambda row: (-row["mrr@10"], -row["ndcg@10"], row["mu"]))
summary = {
    "selection_metric": "mrr@10",
    "validation_only": True,
    "test_split_locked": True,
    "best_mu": rows[0]["mu"],
    "results_ranked": rows,
}
summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
print(json.dumps(summary, indent=2))
PY

echo "0" >"$SWEEP_DONE"
echo "[B4-sweep] completed at $(date -u +%FT%TZ)"
