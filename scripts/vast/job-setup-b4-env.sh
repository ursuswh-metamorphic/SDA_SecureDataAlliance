#!/bin/bash
set -euo pipefail

LOG=/workspace/sda-setup-b4.log
DONE=/workspace/sda-setup-b4.done
rm -f "$DONE"
exec >>"$LOG" 2>&1

echo "[setup-b4] started at $(date -u +%FT%TZ)"
python3 -m venv /workspace/sda-venv
/usr/local/bin/uv pip install \
  --python /workspace/sda-venv/bin/python \
  torch torchvision transformers peft numpy pytest rapidfuzz \
  requests scipy matplotlib prettytable ujson pyyaml pynvml \
  accelerate sentencepiece

/workspace/sda-venv/bin/python -c \
  "import torch, transformers, peft; assert torch.cuda.is_available(); x=torch.ones(1, device='cuda'); print(torch.__version__, transformers.__version__, peft.__version__, torch.cuda.get_device_name(0), x)"

cd /workspace/SDA_SecureDataAlliance
/workspace/sda-venv/bin/python -m pytest \
  FedE/tests/test_fedprox.py \
  FedE/tests/test_weighted_fedavg.py \
  FedE/tests/test_pooling_consistency.py -q

echo "0" >"$DONE"
echo "[setup-b4] completed at $(date -u +%FT%TZ)"
