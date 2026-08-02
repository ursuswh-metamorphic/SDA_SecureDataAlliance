#!/bin/bash
set -euo pipefail

ROOT="${ROOT:-/workspace/SDA_SecureDataAlliance}"
PYTHON="${PYTHON:-/workspace/sda-venv/bin/python}"
DATASET="DocAILab/FedE4RAG_Dataset"
REVISION="${REVISION:-398304846743f184d36f2c35a3db58fa9be70a9d}"
DEST="$ROOT/artifacts/data/paper_faithful_source"
SOURCE="$DEST/FEDE4FIN"
MANIFEST="$ROOT/artifacts/data/paper_faithful/manifest.json"

if [[ ! -x "$PYTHON" ]]; then
  PYTHON="$(command -v python3)"
fi

HF_CLI="${HF_CLI:-/workspace/sda-venv/bin/hf}"
if [[ ! -x "$HF_CLI" ]]; then
  HF_CLI="$(command -v hf)"
fi

mkdir -p "$DEST" "$(dirname "$MANIFEST")"

# Use separate downloads because some huggingface_hub CLI versions only honor
# the final --include argument when it is repeated.
"$HF_CLI" download "$DATASET" \
  --repo-type dataset \
  --revision "$REVISION" \
  --include 'FEDE4FIN/train_data/data_50000_random.json' \
  --local-dir "$DEST"

"$HF_CLI" download "$DATASET" \
  --repo-type dataset \
  --revision "$REVISION" \
  --include 'FEDE4FIN/train_corpus.json' \
  --local-dir "$DEST"

"$PYTHON" "$ROOT/FedE/tools/validate_paper_faithful_data.py" \
  --train "$SOURCE/train_data/data_50000_random.json" \
  --train-corpus "$SOURCE/train_corpus.json" \
  --output "$MANIFEST"

echo "Paper-faithful data is ready. Manifest: $MANIFEST"
