#!/bin/bash
set -euo pipefail

ROOT=/workspace/SDA_SecureDataAlliance
PYTHON=/workspace/sda-venv/bin/python
LOG=/workspace/sda-setup-b4.log
DONE=/workspace/sda-setup-b4.done

exec >>"$LOG" 2>&1
echo "[setup-b4] installing missing project dependencies at $(date -u +%FT%TZ)"

/usr/local/bin/uv pip install \
  --python "$PYTHON" \
  requests scipy matplotlib prettytable ujson pyyaml pynvml \
  accelerate sentencepiece

"$PYTHON" -c \
  "import torch, transformers, peft; assert torch.cuda.is_available(); print(torch.__version__, transformers.__version__, peft.__version__, torch.cuda.get_device_name(0))"

cd "$ROOT"
"$PYTHON" -m pytest \
  FedE/tests/test_fedprox.py \
  FedE/tests/test_weighted_fedavg.py \
  FedE/tests/test_pooling_consistency.py -q

echo "0" >"$DONE"
echo "[setup-b4] completed at $(date -u +%FT%TZ)"

exec /workspace/job-b4-fedprox-full-smoke.sh
