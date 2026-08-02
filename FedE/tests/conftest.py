"""Test-process environment for the PyTorch-only FedE suite."""

import os
import sys
from pathlib import Path


FEDE_DIR = Path(__file__).resolve().parents[1]
if str(FEDE_DIR) not in sys.path:
    sys.path.insert(0, str(FEDE_DIR))

# transformers should not probe optional TensorFlow/vision stacks while these
# text-retrieval tests import the model code.
os.environ.setdefault("USE_TF", "0")
os.environ.setdefault("TRANSFORMERS_NO_TF", "1")

# ranx imports ir_datasets, which otherwise writes below the user's home.
# Keep its cache inside the project test cache for sandboxed/reproducible runs.
os.environ.setdefault(
    "IR_DATASETS_HOME",
    str(FEDE_DIR / ".pytest_cache" / "ir_datasets"),
)
os.environ.setdefault(
    "IR_DATASETS_TMP",
    str(FEDE_DIR / ".pytest_cache" / "ir_datasets_tmp"),
)
