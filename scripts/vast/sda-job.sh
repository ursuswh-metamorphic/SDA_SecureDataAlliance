#!/bin/bash
set -euo pipefail

JOB_FILE=/workspace/sda-current-job.sh
if [[ ! -x "$JOB_FILE" ]]; then
  echo "[sda_job] missing executable job file: $JOB_FILE"
  exit 2
fi

echo "[sda_job] starting $JOB_FILE at $(date -u +%FT%TZ)"
exec "$JOB_FILE"
