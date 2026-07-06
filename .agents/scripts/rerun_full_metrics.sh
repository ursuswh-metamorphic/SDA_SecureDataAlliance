#!/bin/bash
# =============================================================================
# rerun_full_metrics.sh — re-evaluate ALL checkpoints with the full paper-parity
# metric set (Hit@1/3/5/10, MRR, MAP, NDCG@10, EM, Recall/Precision@{1,5,10}).
#
# EVAL-ONLY (no training). Cheap: ~1 min/eval on GPU → ~20 min for everything.
# Uses the updated eval_paper_protocol.py (commit 78ca478+). Requires the
# checkpoints to be present in FedE/ (upload them first — see below).
#
# ---- UPLOAD checkpoints to the server first (from repo root, local) ---------
#   scp -P <PORT> \
#     .agents/validation_artifacts_phase7C/checkpoints/dp_paper_qa_eps1_final.bin \
#     .agents/validation_artifacts_phase7C/checkpoints/dp_paper_qa_eps50_final.bin \
#     .agents/validation_artifacts_phase7C/checkpoints/dp_paper_qa_eps100_final.bin \
#     .agents/validation_artifacts_phase6.5D/checkpoints/dp_lora_paper_qa_final.bin \
#     .agents/validation_artifacts_khung1_ablation/checkpoints/ablate_nodp_noKD.bin \
#     .agents/validation_artifacts_khung1_ablation/checkpoints/ablate_nodp_fullft.bin \
#     .agents/validation_artifacts_khung1_ablation/checkpoints/ablate_dp_noKD.bin \
#     .agents/validation_artifacts_khung1_ablation/checkpoints/ablate_dp_fullft.bin \
#     root@<IP>:/root/sda/FedE/
#
# PREREQUISITE: same setup as the ε-sweep (venv + torch + transformers + peft +
#   huggingface_hub + llama-index-core; FedE/scripts/download_paper_data.py run).
#
# Usage (from repo root):  bash .agents/scripts/rerun_full_metrics.sh
# =============================================================================
set -uo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
VENV="${VENV:-$REPO_ROOT/venv}"
BATCH="${BATCH:-64}"

cd "$REPO_ROOT/FedE"
# Activate the venv — search common locations ($VENV override, /root/venv from
# the quickstart setup, or repo_root/../venv). shellcheck disable=SC1090
for _v in "$VENV" /root/venv "$REPO_ROOT/../venv"; do
  [ -f "$_v/bin/activate" ] && { source "$_v/bin/activate"; echo "[venv] $_v"; break; }
done
export PYTHONUNBUFFERED=1

# name : checkpoint file ('' = pretrained baseline, no checkpoint)
CONFIGS=(
  "pretrained          "
  "dp_eps1             dp_paper_qa_eps1_final.bin"
  "dp_eps20            dp_lora_paper_qa_final.bin"
  "dp_eps50            dp_paper_qa_eps50_final.bin"
  "dp_eps100           dp_paper_qa_eps100_final.bin"
  "ablate_nodp_noKD    ablate_nodp_noKD.bin"
  "ablate_nodp_fullft  ablate_nodp_fullft.bin"
  "ablate_dp_noKD      ablate_dp_noKD.bin"
  "ablate_dp_fullft    ablate_dp_fullft.bin"
)

for cfg in "${CONFIGS[@]}"; do
  # shellcheck disable=SC2086
  set -- $cfg; name="$1"; ckpt="${2:-}"
  if [ -n "$ckpt" ] && [ ! -f "$ckpt" ]; then
    echo "### SKIP $name — checkpoint $ckpt not uploaded"; continue
  fi
  for split in val test; do
    echo "############## EVAL $name / $split  $(date +%T) ##############"
    if [ -z "$ckpt" ]; then
      python -X utf8 eval_paper_protocol.py --use-llama-index --name "${name}_fullmetrics" \
        --split "$split" --batch-size "$BATCH" 2>&1 | grep -E "\[Presence\]|\[Order\]|\[Threshold\]|results:"
    else
      python -X utf8 eval_paper_protocol.py --checkpoint "$ckpt" --use-llama-index \
        --name "${name}_fullmetrics" --split "$split" --batch-size "$BATCH" 2>&1 | grep -E "\[Presence\]|\[Order\]|\[Threshold\]|results:"
    fi
  done
done
touch /root/rerun_metrics.done
echo "########## FULL-METRIC RE-EVAL DONE  $(date) ##########"
echo "JSONs: FedE/paper_test_data/eval_outputs/paper_protocol_*_fullmetrics_*.json"
