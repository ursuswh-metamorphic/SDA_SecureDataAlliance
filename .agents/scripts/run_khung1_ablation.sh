#!/bin/bash
# =============================================================================
# run_khung1_ablation.sh — retriever recipe ablation for Khung đánh giá 1.
#
# Trains + evals the recipe-ablation cells:
#   C   ablate_dp_noKD     : DP + LoRA, KD-GLE OFF        (DP=1 KD=0 LORA=1)
#   C'  ablate_nodp_noKD   : Non-DP + LoRA, KD OFF        (DP=0 KD=0 LORA=1)
#   D   ablate_dp_fullft   : DP + FULL fine-tune (−PEFT)  (DP=1 KD=1 LORA=0)
#   D'  ablate_nodp_fullft : Non-DP + FULL fine-tune      (DP=0 KD=1 LORA=0)
#   (optional, WITH_QLORA=1)
#   ablate_dp_qlora / ablate_nodp_qlora : qLoRA variant   (QLORA=1)
#
# Already measured elsewhere (do NOT re-run): A = DP+LoRA+KD (ε-sweep, 62/62),
# B = nonDP+LoRA+KD (Run 10, 58/56), Pretrained (62/62), DP-strength (ε-sweep).
# Full fine-tune (D/D') uses USE_LORA=0 — the no-PEFT path added to
# config.get_model + fedrag_lora (transports full state; ~436MB checkpoints).
# NOTE: full-FT is heavier (per-sample DP over 109M params) — watch round 0.
#
# PREREQUISITE: same env as the ε-sweep (git clone + venv + torch + transformers
#   + peft + huggingface_hub + llama-index-core; FedE/scripts/download_paper_data.py;
#   selected_data.json present). Run FROM repo root on a GPU server.
#
# Usage:  bash .agents/scripts/run_khung1_ablation.sh
#         WITH_QLORA=1 bash .agents/scripts/run_khung1_ablation.sh   # + qLoRA cells
# =============================================================================
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
VENV="${VENV:-$REPO_ROOT/venv}"
BATCH="${BATCH:-64}"

# ---- CPU sanity gate (avoid the throttled-server bottleneck) ----------------
MHZ=$(grep -i 'cpu MHz' /proc/cpuinfo | awk '{print $4}' | sort -rn | head -1 | cut -d. -f1 || echo 0)
echo "[precheck] cpu MHz(max)=$MHZ  nproc=$(nproc)"
if [ "${MHZ:-0}" -lt 2500 ]; then
  echo "❌ ABORT: cpu MHz $MHZ < 2500 → per-sample DP will be ~10× slow. Switch server."; exit 1
fi
python3 - <<'PY' || { echo "❌ ABORT: single-thread CPU too slow (throttled?)."; exit 1; }
import time,sys
t=time.time(); s=0
for i in range(10_000_000): s+=i
print(f"[precheck] single-thread bench: {time.time()-t:.2f}s (expect < 1.2s)")
sys.exit(1 if time.time()-t>1.2 else 0)
PY
echo "[precheck] CPU OK ✓"

cd "$REPO_ROOT/FedE"
# shellcheck disable=SC1090
[ -d "$VENV" ] && source "$VENV/bin/activate"
export PYTHONUNBUFFERED=1
[ -f selected_data.json ] || { echo "❌ selected_data.json missing"; exit 1; }
[ -d paper_test_data ]    || { echo "❌ paper_test_data/ missing"; exit 1; }

# tag  DP  KD  QLORA  LORA
CELLS=(
  "ablate_dp_noKD     1 0 0 1"   # C   : DP + LoRA, KD-GLE off
  "ablate_nodp_noKD   0 0 0 1"   # C'  : Non-DP + LoRA, KD-GLE off
  "ablate_dp_fullft   1 1 0 0"   # D   : DP + FULL fine-tune (−PEFT), KD on
  "ablate_nodp_fullft 0 1 0 0"   # D'  : Non-DP + FULL fine-tune (−PEFT), KD on
)
if [ "${WITH_QLORA:-0}" = 1 ]; then
  CELLS+=("ablate_dp_qlora 1 1 1 1" "ablate_nodp_qlora 0 1 1 1")
fi

for cell in "${CELLS[@]}"; do
  # shellcheck disable=SC2086
  set -- $cell; tag="$1"; dp="$2"; kd="$3"; ql="$4"; lora="$5"
  echo "############## TRAIN $tag  (DP=$dp KD=$kd QLORA=$ql LORA=$lora)  $(date) ##############"
  rm -rf num5_alpha05_lora training.log x-lora_*.bin checkpoints/
  DP_ENABLED="$dp" KD_WEIGHT="$kd" USE_QLORA="$ql" USE_LORA="$lora" TARGET_EPS=20 \
    python -X utf8 -u main_lora.py > "train_${tag}.log" 2>&1 || { echo "train failed $tag"; tail -20 "train_${tag}.log"; continue; }
  ckpt=$(ls -t x-lora_*.bin 2>/dev/null | head -1 || true)
  if [ -z "$ckpt" ]; then echo "⚠ no checkpoint for $tag"; tail -15 "train_${tag}.log"; continue; fi
  mv "$ckpt" "${tag}.bin"
  echo "### trained $tag -> ${tag}.bin  $(date)"
  for split in val test; do
    echo "### EVAL $tag / $split"
    python -X utf8 eval_paper_protocol.py --checkpoint "${tag}.bin" --use-llama-index \
      --name "$tag" --split "$split" --batch-size "$BATCH" 2>&1 | grep -E "Hit@1|Hit@10|MRR|results:"
  done
done
touch /root/khung1_ablation.done
echo "########## KHUNG 1 ABLATION DONE  $(date) ##########"
echo "Result JSONs: FedE/paper_test_data/eval_outputs/paper_protocol_ablate_*.json"
