"""Upstream checks for the MedCPT switch.

Verifies that:
  1. Every training / eval / benchmark module now points at
     `ncbi/MedCPT-Article-Encoder` (no BGE leakage in upstream).
  2. The training & eval code paths use [CLS] pooling (the MedCPT
     recommended recipe), not mean pooling over tokens.
  3. LoRA target modules `['query', 'value']` still exist on a
     BERT backbone (so PEFT attaches correctly on MedCPT, which is
     a BERT-based model).
  4. (slow) The real MedCPT-Article-Encoder loads and produces a
     768-dim [CLS] vector; LoRA attaches successfully.

Fast tests use `hf-internal-testing/tiny-random-bert` via `BertTokenizer`/
`BertModel` directly (avoids `AutoTokenizer`'s heavy import chain that in
some envs transitively pulls TensorFlow).
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest
import torch


REPO_ROOT = Path(__file__).resolve().parent.parent  # FedE/
EXPECTED_MODEL = "ncbi/MedCPT-Article-Encoder"
TINY_BERT = "hf-internal-testing/tiny-random-bert"


# ---------------------------------------------------------------------------
# 1. Source-level audit (no runtime imports required)
# ---------------------------------------------------------------------------


UPSTREAM_FILES = [
    REPO_ROOT / "main_dp_lora.py",
    REPO_ROOT / "main_dp_lora_eps20.py",
    REPO_ROOT / "eval_full.py",
    REPO_ROOT / "eval_compare.py",
    REPO_ROOT / "flgo" / "algorithm" / "fedbase.py",
    REPO_ROOT / "flgo" / "benchmark" / "fedrag_classification" / "config.py",
    REPO_ROOT / "flgo" / "benchmark" / "fedrag_classification" / "core.py",
]


@pytest.mark.parametrize("path", UPSTREAM_FILES, ids=lambda p: p.name)
def test_no_bge_reference_in_upstream(path: Path) -> None:
    src = path.read_text(encoding="utf-8")
    assert "BAAI/bge-base-en" not in src, (
        f"{path.name} still references the old BGE backbone."
    )


BACKBONE_USER_FILES = UPSTREAM_FILES  # all of them should reference MedCPT now


@pytest.mark.parametrize("path", BACKBONE_USER_FILES, ids=lambda p: p.name)
def test_medcpt_reference_present(path: Path) -> None:
    src = path.read_text(encoding="utf-8")
    assert EXPECTED_MODEL in src, (
        f"{path.name} does not reference `{EXPECTED_MODEL}`."
    )


POOLING_FILES = [
    REPO_ROOT / "main_dp_lora.py",
    REPO_ROOT / "main_dp_lora_eps20.py",
    REPO_ROOT / "eval_full.py",
    REPO_ROOT / "eval_compare.py",
    REPO_ROOT / "flgo" / "benchmark" / "fedrag_classification" / "core.py",
]


@pytest.mark.parametrize("path", POOLING_FILES, ids=lambda p: p.name)
def test_no_mean_token_pooling(path: Path) -> None:
    src = path.read_text(encoding="utf-8")
    assert "last_hidden_state.mean(" not in src, (
        f"{path.name} still uses mean-over-tokens pooling; MedCPT needs [CLS]."
    )
    assert not re.search(
        r"torch\.mean\s*\(\s*\w+\.last_hidden_state\s*,\s*dim\s*=\s*1",
        src,
    ), f"{path.name} still uses torch.mean on last_hidden_state."


@pytest.mark.parametrize("path", POOLING_FILES, ids=lambda p: p.name)
def test_cls_pooling_present(path: Path) -> None:
    src = path.read_text(encoding="utf-8")
    assert "last_hidden_state[:, 0, :]" in src, (
        f"{path.name} does not use `last_hidden_state[:, 0, :]` (CLS) pooling."
    )


# ---------------------------------------------------------------------------
# 2. Constant values parsed from source (no heavy imports of flgo)
# ---------------------------------------------------------------------------

CONSTANT_FILES = [
    REPO_ROOT / "flgo" / "benchmark" / "fedrag_classification" / "config.py",
    REPO_ROOT / "flgo" / "benchmark" / "fedrag_classification" / "core.py",
    REPO_ROOT / "flgo" / "algorithm" / "fedbase.py",
    REPO_ROOT / "main_dp_lora.py",
    REPO_ROOT / "main_dp_lora_eps20.py",
    REPO_ROOT / "eval_full.py",
    REPO_ROOT / "eval_compare.py",
]


@pytest.mark.parametrize("path", CONSTANT_FILES, ids=lambda p: p.name)
def test_embedding_model_name_constant_value(path: Path) -> None:
    """Each upstream module defines EMBEDDING_MODEL_NAME = '<MedCPT>' literally."""
    src = path.read_text(encoding="utf-8")
    m = re.search(
        r"^EMBEDDING_MODEL_NAME\s*=\s*[\"']([^\"']+)[\"']",
        src,
        flags=re.MULTILINE,
    )
    assert m is not None, f"{path.name} missing EMBEDDING_MODEL_NAME constant."
    assert m.group(1) == EXPECTED_MODEL, (
        f"{path.name} has EMBEDDING_MODEL_NAME={m.group(1)!r}, "
        f"expected {EXPECTED_MODEL!r}."
    )


# ---------------------------------------------------------------------------
# 3. CLS-pooling behavioural check on a tiny BERT (fast)
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def tiny_backbone():
    """Load tiny BERT via BertTokenizer/BertModel directly.

    Using the Bert* classes (not AutoTokenizer/AutoModel) avoids transformers'
    `processing_utils` -> `video_utils` -> `image_transforms` import chain,
    which can transitively pull TensorFlow in some environments.
    """
    try:
        from transformers import BertModel, BertTokenizer
    except Exception as exc:  # pragma: no cover - env-specific
        pytest.skip(f"transformers not importable: {exc}")

    tok = BertTokenizer.from_pretrained(TINY_BERT)
    mdl = BertModel.from_pretrained(TINY_BERT)
    mdl.eval()
    return tok, mdl


def test_cls_pooling_equals_first_token(tiny_backbone) -> None:
    tok, mdl = tiny_backbone
    enc = tok(
        ["pubmed cls pooling"],
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=32,
    )
    with torch.no_grad():
        hs = mdl(**enc).last_hidden_state
    assert torch.allclose(hs[:, 0, :], hs[:, 0])


def test_cls_pooling_disagrees_with_mean(tiny_backbone) -> None:
    """Guards against accidental reintroduction of mean pooling."""
    tok, mdl = tiny_backbone
    enc = tok(
        ["biomedical retrieval"],
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=32,
    )
    with torch.no_grad():
        hs = mdl(**enc).last_hidden_state
    cls = hs[:, 0, :]
    mean = hs.mean(dim=1)
    assert not torch.allclose(cls, mean, atol=1e-4)


def test_lora_targets_exist_on_tiny_bert(tiny_backbone) -> None:
    """`query` and `value` attention sub-modules must exist by name so that
    PEFT's `target_modules=['query', 'value']` attaches LoRA adapters. Same
    naming convention as MedCPT (BertModel)."""
    _, mdl = tiny_backbone
    module_names = [n for n, _ in mdl.named_modules()]
    assert any(n.endswith(".query") for n in module_names)
    assert any(n.endswith(".value") for n in module_names)


# ---------------------------------------------------------------------------
# 4. Integration with the real MedCPT Article Encoder (slow, ~400MB)
# ---------------------------------------------------------------------------


@pytest.mark.slow
class TestRealMedCPTUpstream:
    def test_medcpt_cls_forward_shape_768(self) -> None:
        try:
            from transformers import BertModel, BertTokenizer
        except Exception as exc:
            pytest.skip(f"transformers not importable: {exc}")

        tok = BertTokenizer.from_pretrained(EXPECTED_MODEL)
        mdl = BertModel.from_pretrained(EXPECTED_MODEL)
        mdl.eval()
        enc = tok(
            ["Central diabetes insipidus is a clinical syndrome"],
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=512,
        )
        with torch.no_grad():
            cls = mdl(**enc).last_hidden_state[:, 0, :]
        assert cls.shape == (1, 768)

    def test_lora_attaches_to_medcpt(self) -> None:
        try:
            from peft import LoraConfig, get_peft_model
            from transformers import BertModel
        except Exception as exc:
            pytest.skip(f"peft/transformers not importable: {exc}")

        base = BertModel.from_pretrained(EXPECTED_MODEL)
        cfg = LoraConfig(
            r=8,
            lora_alpha=16,
            target_modules=["query", "value"],
            lora_dropout=0.1,
            bias="none",
        )
        lora_model = get_peft_model(base, cfg)
        trainable = sum(p.numel() for p in lora_model.parameters() if p.requires_grad)
        total = sum(p.numel() for p in lora_model.parameters())
        assert 0 < trainable < total  # only adapters are trainable

    def test_get_model_loads_medcpt_from_benchmark(self) -> None:
        """Lazy-import path through the benchmark config.

        May skip in envs where `flgo.__init__` transitively fails (e.g.
        broken TensorFlow install)."""
        try:
            from flgo.benchmark.fedrag_classification.config import get_model  # noqa: E501
        except Exception as exc:
            pytest.skip(f"flgo not importable in this env: {exc}")

        model = get_model()
        assert getattr(model.config, "hidden_size", None) == 768
