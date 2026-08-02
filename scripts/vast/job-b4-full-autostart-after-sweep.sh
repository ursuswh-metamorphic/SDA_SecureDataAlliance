#!/bin/bash
set -euo pipefail

ROOT=/workspace/SDA_SecureDataAlliance
SWEEP_DONE="$ROOT/artifacts/training/b4_fedprox_mu_sweep_seed13/sweep.done"
SWEEP_LOG="$ROOT/artifacts/training/b4_fedprox_mu_sweep_seed13/sweep.log"
FULL_JOB=/workspace/job-b4-fedprox-full-pilot-selected.sh
CURRENT_JOB=/workspace/sda-current-job.sh
LOG=/workspace/sda-b4-full-autostart.log

exec >>"$LOG" 2>&1
echo "[b4-full-autostart] waiting for sweep at $(date -u +%FT%TZ)"

while supervisorctl status sda_job 2>/dev/null | grep -q RUNNING; do
  sleep 20
done

if [[ ! -f "$SWEEP_DONE" ]] || [[ "$(cat "$SWEEP_DONE")" != "0" ]]; then
  echo "[b4-full-autostart] sweep did not complete successfully"
  tail -n 100 "$SWEEP_LOG" || true
  exit 1
fi

install -m 0755 "$FULL_JOB" "$CURRENT_JOB"
supervisorctl start sda_job
echo "[b4-full-autostart] selected-mu full pilot submitted at $(date -u +%FT%TZ)"
