#!/bin/bash
# =============================================================================
# complete_eps_sweep.sh — finish the Phase 7C privacy–utility (ε) sweep.
#
# Does, in order:
#   1. CPU sanity gate  — abort if the server is throttled (the Phase-7C lesson:
#      the per-sample DP loop is single-thread-CPU-bound; a cgroup-throttled
#      box was 10× slower → 108h instead of 2h).
#   2. Eval ε=1         — checkpoint already trained locally, just needs the
#      measured retrieval numbers (LlamaIndex protocol, val + test).
#   3. Train + eval ε=50 and ε=100  — from scratch (previous runs were partial
#      / failed). σ auto-calibrated by main_lora.py via TARGET_EPS.
#   4. Print where the result JSONs landed.
#
# ε=20 (dp_lora_paper_qa_final.bin) and pretrained are ALREADY done (62/62) —
# not re-run here. Add them to SETUPS below if you want a fully self-contained
# table.
#
# ---- PREREQUISITES (run FROM repo root on the GPU server) -------------------
#   git clone -b feature/validate_old_data ... sda && cd sda
#   python3 -m venv venv && source venv/bin/activate
#   pip install torch --index-url https://download.pytorch.org/whl/cu124
#   pip install transformers peft huggingface_hub numpy requests ujson scipy \
#               prettytable pynvml llama-index llama-index-embeddings-huggingface
#   cd FedE && python scripts/download_paper_data.py        # test corpus + qa
#   # training data:
#   python -c "from huggingface_hub import hf_hub_download; import shutil; \
#     p=hf_hub_download('DocAILab/FedE4RAG_Dataset','FEDE4FIN/train_data/data_50000_random.json',repo_type='dataset'); \
#     shutil.copy(p,'selected_data.json')"
#   # upload the ε=1 checkpoint (skips a 2h retrain):
#   #   scp .agents/validation_artifacts_phase7C/checkpoints/dp_paper_qa_eps1_final.bin \
#   #       <server>:/root/sda/FedE/
#
# Usage (from repo root):  bash .agents/scripts/complete_eps_sweep.sh
# =============================================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
VENV="${VENV:-$REPO_ROOT/venv}"
BATCH="${BATCH:-64}"

# ---- 1. CPU sanity gate -----------------------------------------------------
MHZ=$(grep -i 'cpu MHz' /proc/cpuinfo | awk '{print $4}' | sort -rn | head -1 | cut -d. -f1 || echo 0)
CORES=$(nproc)
echo "[precheck] cpu MHz(max under load)=$MHZ   nproc=$CORES"
if [ "${MHZ:-0}" -lt 2500 ]; then
  echo "❌ ABORT: cpu MHz $MHZ < 2500 → per-sample DP will be ~10× slow."
  echo "   Rent a faster box (EPYC 9xxx / Xeon Gold / Threadripper PRO). See HANDOFF_phase7C."
  exit 1
fi
python3 - <<'PY' || { echo "❌ ABORT: single-thread CPU too slow (throttled?). Switch server."; exit 1; }
import time, sys
t=time.time(); s=0
for i in range(10_000_000): s+=i
d=time.time()-t
print(f"[precheck] single-thread bench: {d:.2f}s (expect < 0.6s on a healthy box)")
sys.exit(1 if d > 1.2 else 0)
PY
echo "[precheck] CPU OK ✓"

cd "$REPO_ROOT/FedE"
# shellcheck disable=SC1090
[ -d "$VENV" ] && source "$VENV/bin/activate"
export PYTHONUNBUFFERED=1

[ -f selected_data.json ] || { echo "❌ selected_data.json missing (see prerequisites)"; exit 1; }
[ -d paper_test_data ]    || { echo "❌ paper_test_data/ missing (run download_paper_data.py)"; exit 1; }

eval_ckpt () {  # $1=checkpoint(file or '' for pretrained)  $2=name
  local ckpt="$1" name="$2"
  for split in val test; do
    echo "=== EVAL $name / $split ==="
    if [ -z "$ckpt" ]; then
      python -X utf8 eval_paper_protocol.py --use-llama-index --name "$name" --split "$split" --batch-size "$BATCH"
    else
      python -X utf8 eval_paper_protocol.py --checkpoint "$ckpt" --use-llama-index --name "$name" --split "$split" --batch-size "$BATCH"
    fi
  done
}

# ---- 2. Eval ε=1 (already trained) -----------------------------------------
if [ -f dp_paper_qa_eps1_final.bin ]; then
  eval_ckpt dp_paper_qa_eps1_final.bin dp_eps1
else
  echo "[skip ε=1] dp_paper_qa_eps1_final.bin not uploaded — will (re)train below if you add 1 to EPS_LIST."
fi

# ---- 3. Train + eval ε=50 and ε=100 ----------------------------------------
EPS_LIST="${EPS_LIST:-50 100}"
for eps in $EPS_LIST; do
  echo "############## TRAIN ε=$eps ##############"
  rm -rf num5_alpha05_lora training.log x-lora_*.bin
  DP_ENABLED=1 TARGET_EPS="$eps" python -X utf8 -u main_lora.py 2>&1 | tee "train_eps${eps}.log"
  ckpt=$(ls -t x-lora_*.bin 2>/dev/null | head -1 || true)
  if [ -n "$ckpt" ]; then
    mv "$ckpt" "dp_paper_qa_eps${eps}_final.bin"
    eval_ckpt "dp_paper_qa_eps${eps}_final.bin" "dp_eps${eps}"
  else
    echo "⚠ no x-lora_*.bin produced for ε=$eps — check train_eps${eps}.log"
  fi
done

echo
echo "=== DONE. Result JSONs: FedE/paper_test_data/eval_outputs/paper_protocol_dp_eps*_*.json ==="
echo "    Grep the numbers:  grep -H 'Hit@1\\|MRR' paper_test_data/eval_outputs/paper_protocol_dp_eps*.json"
echo "    Then fill §7 of docs/Federated_RAG_Privacy.xlsx (ε=1/50/100 rows)."
