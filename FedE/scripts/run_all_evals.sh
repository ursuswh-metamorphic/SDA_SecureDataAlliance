#!/bin/bash
# Run paper-faithful eval on all 4 setups × 2 splits = 8 evaluations.
# Corpus encoding (~30K pages) is the slow part — done ONCE per checkpoint.
#
# Usage (from repo root):
#   bash FedE/scripts/run_all_evals.sh
#
# Assumes:
#   - cwd contains FedE/   (run from repo root)
#   - python venv at $VENV (default: ../venv from FedE/, override via env)
#   - 3 trained checkpoints in FedE/: fin_lora_nondp_run5.bin, fin_dp_run4.bin, fin_dp_qlora_run10.bin
#   - paper_test_data/ populated (run download_paper_data.py first)
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
FEDE_DIR="$(dirname "$SCRIPT_DIR")"
VENV="${VENV:-$(dirname "$FEDE_DIR")/venv}"

cd "$FEDE_DIR"
[ -d "$VENV" ] && source "$VENV/bin/activate"

declare -A SETUPS=(
  [pretrained]=""
  [non_dp_lora]="fin_lora_nondp_run5.bin"
  [dp_lora_eps20]="fin_dp_run4.bin"
  [dp_qlora_eps20]="fin_dp_qlora_run10.bin"
)

echo "=== Eval 4 setups × 2 splits = 8 runs starting at $(date) ==="
total_start=$(date +%s)
for setup in pretrained non_dp_lora dp_lora_eps20 dp_qlora_eps20; do
  ckpt="${SETUPS[$setup]}"
  for split in val test; do
    echo
    echo "=== EVAL: $setup / $split ==="
    if [ -z "$ckpt" ]; then
      python -X utf8 eval_paper_faithful.py --split $split --name $setup
    else
      python -X utf8 eval_paper_faithful.py --checkpoint $ckpt --split $split --name $setup
    fi
  done
done
total_end=$(date +%s)
echo
echo "=== ALL 8 EVALS DONE in $((total_end - total_start)) seconds ==="
ls -la paper_test_data/eval_outputs/
