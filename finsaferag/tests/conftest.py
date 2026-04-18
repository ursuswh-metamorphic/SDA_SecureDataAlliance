"""Pytest config for finsaferag tests.

- Sets env vars BEFORE any transformers import, to keep transformers from
  transitively importing TensorFlow (which can crash on NumPy>=2 envs).
- Adds finsaferag/finsaferag/ to sys.path so siblings like
  `embs.embedding` can be imported directly (matching runtime layout).
- Registers the `slow` marker for tests that pull real MedCPT weights
  from Hugging Face (~400MB). Skipped by default.
"""

from __future__ import annotations

import os
import sys

# Must be set before `transformers` / `torch` are imported anywhere.
os.environ.setdefault("USE_TF", "0")
os.environ.setdefault("TRANSFORMERS_NO_ADVISORY_WARNINGS", "1")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")

import pytest  # noqa: E402

_PKG_DIR = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "finsaferag")
)
if _PKG_DIR not in sys.path:
    sys.path.insert(0, _PKG_DIR)


def pytest_configure(config: pytest.Config) -> None:
    config.addinivalue_line(
        "markers",
        "slow: downloads real MedCPT weights from HF (~400MB). Use -m slow to run.",
    )


def pytest_collection_modifyitems(
    config: pytest.Config, items: list[pytest.Item]
) -> None:
    selected = config.getoption("-m") or ""
    if "slow" in selected:
        return
    skip_slow = pytest.mark.skip(
        reason="skipped by default (downloads MedCPT). Run with `-m slow` to enable."
    )
    for item in items:
        if "slow" in item.keywords:
            item.add_marker(skip_slow)
