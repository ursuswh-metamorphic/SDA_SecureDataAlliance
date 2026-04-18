"""Unit & integration tests for the downstream `MedCPTEmbedding` wrapper.

Fast tests use `hf-internal-testing/tiny-random-bert` (a few MB) to exercise
the LlamaIndex `BaseEmbedding` interface and [CLS] pooling semantics without
pulling heavy weights.

Slow tests (decorated with `@pytest.mark.slow`) pull the real
`ncbi/MedCPT-Article-Encoder` and verify dimensional correctness.
Run them with:

    pytest finsaferag/tests -m slow

The module is skipped entirely in environments where `llama_index` is not
installed, since `MedCPTEmbedding` inherits from LlamaIndex's `BaseEmbedding`.
"""

from __future__ import annotations

import math
from typing import Iterable

import pytest

pytest.importorskip(
    "llama_index.core.embeddings",
    reason="llama-index-core is required for MedCPTEmbedding tests.",
)
torch = pytest.importorskip("torch")

# Use Bert* directly to avoid transformers' Auto* -> processing_utils chain
# which can transitively pull TensorFlow in some environments.
_tfm = pytest.importorskip("transformers")
from transformers import BertModel, BertTokenizer  # noqa: E402

from embs.embedding import (  # noqa: E402
    DEFAULT_EMBEDDING_MAX_LENGTH,
    DEFAULT_EMBEDDING_MODEL,
    MedCPTEmbedding,
)


TINY_BERT = "hf-internal-testing/tiny-random-bert"


def _l2(xs: Iterable[float]) -> float:
    return math.sqrt(sum(float(x) * float(x) for x in xs))


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def tiny_emb() -> MedCPTEmbedding:
    return MedCPTEmbedding(model_name=TINY_BERT, max_length=32, device="cpu")


@pytest.fixture(scope="module")
def tiny_raw():
    tok = BertTokenizer.from_pretrained(TINY_BERT)
    mdl = BertModel.from_pretrained(TINY_BERT)
    mdl.eval()
    return tok, mdl


# ---------------------------------------------------------------------------
# Class & interface
# ---------------------------------------------------------------------------


class TestInterface:
    def test_class_name(self, tiny_emb: MedCPTEmbedding) -> None:
        assert tiny_emb.class_name() == "MedCPTEmbedding"

    def test_default_constants(self) -> None:
        assert DEFAULT_EMBEDDING_MODEL == "ncbi/MedCPT-Article-Encoder"
        assert DEFAULT_EMBEDDING_MAX_LENGTH == 512

    def test_query_embedding_returns_list_of_floats(
        self, tiny_emb: MedCPTEmbedding
    ) -> None:
        emb = tiny_emb._get_query_embedding("diabetes insipidus")
        assert isinstance(emb, list)
        assert len(emb) > 0
        assert all(isinstance(x, float) for x in emb)

    def test_text_and_query_share_embedding_dim(
        self, tiny_emb: MedCPTEmbedding
    ) -> None:
        q = tiny_emb._get_query_embedding("hello")
        t = tiny_emb._get_text_embedding("hello")
        assert len(q) == len(t)

    def test_public_high_level_api(self, tiny_emb: MedCPTEmbedding) -> None:
        """The LlamaIndex public surface (BaseEmbedding) must work."""
        q = tiny_emb.get_query_embedding("hello")
        t = tiny_emb.get_text_embedding("hello")
        assert isinstance(q, list) and isinstance(t, list)
        assert len(q) == len(t)


# ---------------------------------------------------------------------------
# Batching
# ---------------------------------------------------------------------------


class TestBatching:
    def test_multi_text_batch(self, tiny_emb: MedCPTEmbedding) -> None:
        texts = ["hello world", "biomedical retrieval", "pubmed articles"]
        embs = tiny_emb._get_text_embeddings(texts)
        assert len(embs) == len(texts)
        assert len({len(e) for e in embs}) == 1

    def test_empty_batch(self, tiny_emb: MedCPTEmbedding) -> None:
        assert tiny_emb._get_text_embeddings([]) == []

    def test_batch_matches_single(self, tiny_emb: MedCPTEmbedding) -> None:
        """Batched and single-item forward must agree."""
        txt = "medcpt cls test"
        single = tiny_emb._get_text_embedding(txt)
        batched = tiny_emb._get_text_embeddings([txt])[0]
        assert len(single) == len(batched)
        diffs = [abs(a - b) for a, b in zip(single, batched)]
        assert max(diffs) < 1e-4


# ---------------------------------------------------------------------------
# Pooling semantics (core MedCPT-compatibility check)
# ---------------------------------------------------------------------------


class TestPooling:
    def test_cls_differs_from_mean(self, tiny_raw) -> None:
        tok, mdl = tiny_raw
        enc = tok(
            ["pubmed diabetes insipidus"],
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

    def test_wrapper_uses_cls_pooling(
        self, tiny_emb: MedCPTEmbedding, tiny_raw
    ) -> None:
        """MedCPTEmbedding output == `last_hidden_state[:, 0, :]` + L2-norm."""
        tok, mdl = tiny_raw
        text = "medcpt cls pooling test"
        enc = tok(
            [text],
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=32,
        )
        with torch.no_grad():
            cls = mdl(**enc).last_hidden_state[:, 0, :]
        cls_norm = torch.nn.functional.normalize(cls, p=2, dim=-1)[0].tolist()

        ours = tiny_emb._get_text_embedding(text)
        assert len(ours) == len(cls_norm)
        diffs = [abs(a - b) for a, b in zip(cls_norm, ours)]
        assert max(diffs) < 1e-4

    def test_wrapper_does_not_use_mean_pooling(
        self, tiny_emb: MedCPTEmbedding, tiny_raw
    ) -> None:
        """Guard against accidental regression to mean pooling."""
        tok, mdl = tiny_raw
        text = "mean pooling must NOT match"
        enc = tok(
            [text], return_tensors="pt", padding=True, truncation=True, max_length=32
        )
        with torch.no_grad():
            mean = mdl(**enc).last_hidden_state.mean(dim=1)
        mean_norm = torch.nn.functional.normalize(mean, p=2, dim=-1)[0].tolist()

        ours = tiny_emb._get_text_embedding(text)
        diffs = [abs(a - b) for a, b in zip(mean_norm, ours)]
        assert max(diffs) > 1e-3


# ---------------------------------------------------------------------------
# Normalization
# ---------------------------------------------------------------------------


class TestNormalization:
    def test_default_normalize_is_unit_vector(
        self, tiny_emb: MedCPTEmbedding
    ) -> None:
        emb = tiny_emb._get_text_embedding("normalize me")
        assert abs(_l2(emb) - 1.0) < 1e-4

    def test_normalize_disabled(self) -> None:
        emb_model = MedCPTEmbedding(
            model_name=TINY_BERT, max_length=32, device="cpu", normalize=False
        )
        emb = emb_model._get_text_embedding("no norm")
        norm = _l2(emb)
        assert abs(norm - 1.0) > 1e-3  # unlikely to land exactly on unit sphere

    def test_batch_all_unit_norm(self, tiny_emb: MedCPTEmbedding) -> None:
        texts = ["a", "bb", "ccc long-ish text token token"]
        embs = tiny_emb._get_text_embeddings(texts)
        for e in embs:
            assert abs(_l2(e) - 1.0) < 1e-4


# ---------------------------------------------------------------------------
# Determinism
# ---------------------------------------------------------------------------


class TestDeterminism:
    def test_same_input_same_output(self, tiny_emb: MedCPTEmbedding) -> None:
        text = "the same input should give the same embedding"
        a = tiny_emb._get_text_embedding(text)
        b = tiny_emb._get_text_embedding(text)
        diffs = [abs(x - y) for x, y in zip(a, b)]
        assert max(diffs) < 1e-6

    def test_different_inputs_differ(self, tiny_emb: MedCPTEmbedding) -> None:
        a = tiny_emb._get_text_embedding("diabetes insipidus")
        b = tiny_emb._get_text_embedding("nephrogenic polyuria")
        assert a != b


# ---------------------------------------------------------------------------
# Async wrappers
# ---------------------------------------------------------------------------


class TestAsync:
    def test_aget_query_matches_sync(self, tiny_emb: MedCPTEmbedding) -> None:
        import asyncio

        text = "async parity"
        sync = tiny_emb._get_query_embedding(text)
        async_result = asyncio.run(tiny_emb._aget_query_embedding(text))
        assert sync == async_result


# ---------------------------------------------------------------------------
# Integration: real MedCPT Article Encoder (slow)
# ---------------------------------------------------------------------------


@pytest.mark.slow
class TestRealMedCPT:
    """Validates the wrapper on actual MedCPT weights. ~400MB download."""

    @pytest.fixture(scope="class")
    def medcpt(self) -> MedCPTEmbedding:
        return MedCPTEmbedding(
            model_name=DEFAULT_EMBEDDING_MODEL,
            max_length=512,
            device="cpu",
        )

    def test_embedding_dim_is_768(self, medcpt: MedCPTEmbedding) -> None:
        emb = medcpt._get_text_embedding(
            "Central diabetes insipidus is a clinical syndrome"
        )
        assert len(emb) == 768

    def test_cls_not_zero(self, medcpt: MedCPTEmbedding) -> None:
        emb = medcpt._get_text_embedding("pubmed retrieval test")
        assert any(abs(x) > 1e-6 for x in emb)

    def test_cls_match_hf_card_pattern(self, medcpt: MedCPTEmbedding) -> None:
        """Reproduce HF card recipe: BertModel(...).last_hidden_state[:, 0, :]."""
        text = "Adipsic diabetes insipidus is a rare disorder"
        tok = BertTokenizer.from_pretrained(DEFAULT_EMBEDDING_MODEL)
        mdl = BertModel.from_pretrained(DEFAULT_EMBEDDING_MODEL)
        mdl.eval()
        enc = tok(
            [text], return_tensors="pt", padding=True, truncation=True, max_length=512
        )
        with torch.no_grad():
            cls = mdl(**enc).last_hidden_state[:, 0, :]
        cls_norm = torch.nn.functional.normalize(cls, p=2, dim=-1)[0].tolist()

        ours = medcpt._get_text_embedding(text)
        diffs = [abs(a - b) for a, b in zip(cls_norm, ours)]
        assert max(diffs) < 1e-4

    def test_semantic_retrieval_sanity(self, medcpt: MedCPTEmbedding) -> None:
        """Biomedical query should be closer to a biomedical passage than to
        an unrelated finance passage."""
        q = medcpt._get_query_embedding("diabetes insipidus treatment")
        bio = medcpt._get_text_embedding(
            "Desmopressin is the standard therapy for central diabetes insipidus."
        )
        fin = medcpt._get_text_embedding(
            "Stock prices of technology companies rose sharply last quarter."
        )

        def cos(a: list[float], b: list[float]) -> float:
            return sum(x * y for x, y in zip(a, b))

        assert cos(q, bio) > cos(q, fin)
