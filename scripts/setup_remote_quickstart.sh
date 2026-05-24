#!/bin/bash
# Setup quick-start cho Vast.ai server mới — FedE4RAG project
#
# Run this on REMOTE (sau khi SSH vào server mới):
#   curl -sL https://raw.githubusercontent.com/ursuswh-metamorphic/SDA_SecureDataAlliance/feature/validate_old_data/.agents/setup_remote_quickstart.sh | bash
# Hoặc scp file này lên rồi: bash setup_remote_quickstart.sh
#
# Total time: ~5-10 min
# Idempotent — chạy nhiều lần OK.

set -e

echo "═══════════════════════════════════════════════════════════════"
echo "  FedE4RAG remote setup — quickstart"
echo "═══════════════════════════════════════════════════════════════"

# ── 1. APT deps ──────────────────────────────────────────────────────
echo
echo "[1/6] Install apt deps..."
apt-get install -y python3-venv git 2>&1 | tail -1

# ── 2. Clone repo (latest commit on feature/validate_old_data) ───────
echo
echo "[2/6] Clone repo branch feature/validate_old_data..."
cd /root
if [ -d sda ]; then
    cd sda && git pull origin feature/validate_old_data 2>&1 | tail -2
else
    git clone -b feature/validate_old_data --depth 1 \
        https://github.com/ursuswh-metamorphic/SDA_SecureDataAlliance.git sda 2>&1 | tail -2
    cd sda
fi
echo "HEAD: $(git log --oneline -1)"

# ── 3. Venv + torch CUDA 12.4 ────────────────────────────────────────
echo
echo "[3/6] Create venv + install torch (CUDA 12.4 build for driver 565+)..."
[ -d venv ] || python3 -m venv venv
source venv/bin/activate
pip install --quiet --upgrade pip
pip install --quiet torch --index-url https://download.pytorch.org/whl/cu124 2>&1 | tail -2

# ── 4. ML + project deps ─────────────────────────────────────────────
echo
echo "[4/6] Install ML deps (transformers, peft, FL/DP/eval)..."
pip install --quiet \
    transformers peft huggingface_hub numpy requests ujson \
    scipy matplotlib prettytable pynvml tenseal accelerate sentencepiece \
    2>&1 | tail -2

echo
echo "[4b] Optional Phase 7A deps: llama-index-core (tokenized chunking)"
pip install --quiet llama-index-core 2>&1 | tail -1 || echo "  llama-index-core fail (optional, can skip)"

# ── 5. Verify CUDA ───────────────────────────────────────────────────
echo
echo "[5/6] Verify torch + CUDA..."
python3 -c "
import torch
print(f'torch={torch.__version__}, cuda={torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'gpu={torch.cuda.get_device_name(0)}, vram={torch.cuda.get_device_properties(0).total_memory/1e9:.1f}GB')
"

# ── 6. Download paper test data + Q-A train data ─────────────────────
echo
echo "[6/6] Download paper data..."
cd FedE

# Paper test data (val + test + corpus, ~138 MB)
if [ ! -f paper_test_data/test_corpus.json ]; then
    python3 scripts/download_paper_data.py 2>&1 | tail -3
else
    echo "  paper_test_data/ already present"
fi

# Paper Q-A training data → selected_data.json (~100 MB)
if [ ! -f selected_data.json ] || [ $(stat -c%s selected_data.json) -lt 90000000 ]; then
    python3 -c "
from huggingface_hub import hf_hub_download
import shutil, os
p = hf_hub_download('DocAILab/FedE4RAG_Dataset', 'FEDE4FIN/train_data/data_50000_random.json', repo_type='dataset')
shutil.copy(p, 'selected_data.json')
print(f'  selected_data.json: {os.path.getsize(\"selected_data.json\")/1024/1024:.1f} MB')
"
else
    echo "  selected_data.json already present ($(du -h selected_data.json | cut -f1))"
fi

echo
echo "═══════════════════════════════════════════════════════════════"
echo "  ✅ SETUP DONE"
echo "═══════════════════════════════════════════════════════════════"
df -h / | tail -1
echo
echo "Next steps depend on what you want to run:"
echo "  • Phase 7C ε=100 train:    DP_ENABLED=1 TARGET_EPS=100 python -X utf8 -u main_lora.py"
echo "  • Eval paper protocol:      python -X utf8 eval_paper_protocol.py --checkpoint X.bin --use-llama-index --split val"
echo "  • See .agents/HANDOFF_phase7C_eps_sweep.md for resume plan"
