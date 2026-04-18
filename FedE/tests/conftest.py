"""Pytest config for FedE upstream tests.

- Sets env vars BEFORE any transformers import, to keep transformers from
  transitively importing TensorFlow (which can crash on NumPy>=2 envs).
- Adds the `FedE/` root to sys.path so benchmark / algorithm modules are
  importable.
- Registers the `slow` marker for tests that download the real MedCPT
  Article Encoder (~400MB). Skipped by default.
"""

from __future__ import annotations

import os
import sys

os.environ.setdefault("USE_TF", "0")
os.environ.setdefault("TRANSFORMERS_NO_ADVISORY_WARNINGS", "1")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")

import pytest  # noqa: E402

_FEDE_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _FEDE_ROOT not in sys.path:
    sys.path.insert(0, _FEDE_ROOT)


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
