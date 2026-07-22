"""ir_metrics.py — standard IR metrics with correct semantics (Giai đoạn B1).

Fixes the P0.2 family:
  * Hit@k truncates the RETRIEVED list (never the golden list).
  * Retrieved IDs are deduped (first occurrence keeps its rank) before any
    metric — duplicates must not double-count.
  * AP@k is standard: mean of precision@rank over relevant ranks, divided by
    min(|qrels|, k).
  * nDCG@k uses binary gains with IDCG computed from the qrel set alone —
    independent of the retrieved list. IDCG is NOT a reported metric.
  * Set-EM survives only as `set_em` for the appendix.

Every function takes (retrieved: list[str], relevant: set[str]) and is
cross-checked against ranx/pytrec_eval in tests/test_ir_metrics.py.

Pure Python, no third-party imports — usable from both the torch env and
.venv-eval.
"""
from __future__ import annotations

import math
from typing import Iterable, Sequence

PRIMARY_METRIC = "mrr@10"          # pre-declared primary endpoint
DEFAULT_KS = (1, 3, 5, 10, 100)


def dedupe(retrieved: Sequence[str]) -> list[str]:
    """Drop duplicate IDs, keeping the FIRST occurrence's rank."""
    seen: set[str] = set()
    out: list[str] = []
    for r in retrieved:
        if r not in seen:
            seen.add(r)
            out.append(r)
    return out


def _require(retrieved: Sequence[str], relevant: Iterable[str]) -> tuple[list[str], set[str]]:
    rel = set(relevant)
    if not rel:
        raise ValueError("empty qrel set — schema validation must reject this "
                         "query before metric computation (fail-fast)")
    return dedupe(retrieved), rel


def hit_at_k(retrieved: Sequence[str], relevant: Iterable[str], k: int) -> float:
    """1.0 iff any of the TOP-K RETRIEVED ids is relevant."""
    r, rel = _require(retrieved, relevant)
    return 1.0 if set(r[:k]) & rel else 0.0


def recall_at_k(retrieved: Sequence[str], relevant: Iterable[str], k: int) -> float:
    r, rel = _require(retrieved, relevant)
    return len(set(r[:k]) & rel) / len(rel)


def precision_at_k(retrieved: Sequence[str], relevant: Iterable[str], k: int) -> float:
    r, rel = _require(retrieved, relevant)
    if k <= 0:
        return 0.0
    return sum(1 for x in r[:k] if x in rel) / k


def mrr_at_k(retrieved: Sequence[str], relevant: Iterable[str], k: int) -> float:
    """1/rank of the first relevant id within top-k; 0.0 if none."""
    r, rel = _require(retrieved, relevant)
    for i, x in enumerate(r[:k]):
        if x in rel:
            return 1.0 / (i + 1)
    return 0.0


def average_precision_at_k(retrieved: Sequence[str], relevant: Iterable[str],
                           k: int) -> float:
    """AP@k, TREC convention (trec_eval `map_cut`, matched by ranx and
    pytrec_eval): sum of precision@rank at each relevant rank within the
    top-k RETRIEVED, divided by the TOTAL number of relevant docs |qrels| —
    not min(|qrels|, k). The two conventions coincide whenever k >= |qrels|
    (always true for our primary k=10 with 1-3 qrels/query)."""
    r, rel = _require(retrieved, relevant)
    hits = 0
    acc = 0.0
    for i, x in enumerate(r[:k]):
        if x in rel:
            hits += 1
            acc += hits / (i + 1)
    return acc / len(rel)


def ndcg_at_k(retrieved: Sequence[str], relevant: Iterable[str], k: int) -> float:
    """Binary-gain nDCG@k. IDCG from the qrel set alone (perfect ranking of
    min(|qrels|, k) relevant docs) — never derived from the retrieved list."""
    r, rel = _require(retrieved, relevant)
    dcg = sum(1.0 / math.log2(i + 2) for i, x in enumerate(r[:k]) if x in rel)
    n_ideal = min(len(rel), k)
    idcg = sum(1.0 / math.log2(i + 2) for i in range(n_ideal))
    return dcg / idcg if idcg > 0 else 0.0


def set_em(retrieved: Sequence[str], relevant: Iterable[str]) -> float:
    """Appendix-only 'Set-EM@|qrels|': top-|qrels| retrieved set == qrel set."""
    r, rel = _require(retrieved, relevant)
    return 1.0 if set(r[:len(rel)]) == rel else 0.0


def compute_all(retrieved: Sequence[str], relevant: Iterable[str],
                ks: Sequence[int] = DEFAULT_KS) -> dict[str, float]:
    """All metrics for one query. Keys like 'hit@10', 'mrr@10', plus 'set_em'."""
    out: dict[str, float] = {}
    for k in ks:
        out[f"hit@{k}"] = hit_at_k(retrieved, relevant, k)
        out[f"recall@{k}"] = recall_at_k(retrieved, relevant, k)
        out[f"precision@{k}"] = precision_at_k(retrieved, relevant, k)
        out[f"mrr@{k}"] = mrr_at_k(retrieved, relevant, k)
        out[f"map@{k}"] = average_precision_at_k(retrieved, relevant, k)
        out[f"ndcg@{k}"] = ndcg_at_k(retrieved, relevant, k)
    out["set_em"] = set_em(retrieved, relevant)
    return out
