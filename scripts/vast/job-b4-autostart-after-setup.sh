#!/bin/bash
set -euo pipefail

SETUP_DONE=/workspace/sda-setup-b4.done
SETUP_LOG=/workspace/sda-setup-b4.log
SMOKE_JOB=/workspace/job-b4-fedprox-full-smoke.sh
CURRENT_JOB=/workspace/sda-current-job.sh
LOG=/workspace/sda-b4-autostart.log

exec >>"$LOG" 2>&1
echo "[b4-autostart] waiting for setup at $(date -u +%FT%TZ)"

while supervisorctl status sda_job 2>/dev/null | grep -q RUNNING; do
  sleep 20
done

if [[ ! -f "$SETUP_DONE" ]] || [[ "$(cat "$SETUP_DONE")" != "0" ]]; then
  echo "[b4-autostart] setup did not complete successfully"
  tail -n 100 "$SETUP_LOG" || true
  exit 1
fi

/workspace/sda-venv/bin/python -c \
  "import torch, transformers, peft; assert torch.cuda.is_available()"
install -m 0755 "$SMOKE_JOB" "$CURRENT_JOB"
supervisorctl start sda_job
echo "[b4-autostart] FedProx smoke submitted at $(date -u +%FT%TZ)"
