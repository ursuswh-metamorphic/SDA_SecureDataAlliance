"""test_ir_metrics.py — toy suite for tools/ir_metrics.py (Giai đoạn B2).

Every case has hand-computed expected values (plan Bảng B2), plus a
property-style cross-check against ranx and pytrec_eval when those libraries
are importable (they live in .venv-eval; the suite still passes without them,
but CI must run at least once in an env where they exist).

Run:  python -m pytest tests/test_ir_metrics.py -v
"""
import math
import os
import sys

import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.abspath(os.path.join(HERE, "..")))

from tools import ir_metrics as m  # noqa: E402


# ── Toy cases from the plan ───────────────────────────────────────────────────

def test_relevant_rank_1():
    run, rel = ["A", "B", "C"], {"A"}
    assert m.hit_at_k(run, rel, 1) == 1.0
    assert m.mrr_at_k(run, rel, 10) == 1.0
    assert m.average_precision_at_k(run, rel, 10) == 1.0
    assert m.ndcg_at_k(run, rel, 10) == 1.0


def test_relevant_rank_3():
    run, rel = ["X", "Y", "A"], {"A"}
    assert m.hit_at_k(run, rel, 1) == 0.0
    assert m.hit_at_k(run, rel, 3) == 1.0
    assert m.mrr_at_k(run, rel, 10) == pytest.approx(1 / 3)


def test_two_qrels_recall():
    run, rel = ["A", "X", "B"], {"A", "B"}
    assert m.recall_at_k(run, rel, 1) == 0.5
    assert m.recall_at_k(run, rel, 3) == 1.0


def test_duplicates_deduped_not_double_counted():
    run, rel = ["A", "A", "B"], {"A", "B"}
    # dedupe -> [A, B]: B sits at rank 2, so recall@2 is full
    assert m.dedupe(run) == ["A", "B"]
    assert m.recall_at_k(run, rel, 2) == 1.0
    # precision@2 counts unique docs only — no double credit for A
    assert m.precision_at_k(run, rel, 2) == 1.0
    # AP: relevant at ranks 1 and 2 -> (1/1 + 2/2) / 2 = 1.0
    assert m.average_precision_at_k(run, rel, 10) == 1.0


def test_hit_truncates_retrieved_not_golden():
    # P0.2 regression: golden list longer than k must NOT be truncated.
    # Relevant doc sits at retrieved rank 3 -> hit@1 = 0 even though
    # golden[0:1] trickery would have made it 1 under the buggy definition.
    run = ["X", "Y", "G2"]
    rel = {"G1", "G2", "G3"}
    assert m.hit_at_k(run, rel, 1) == 0.0
    assert m.hit_at_k(run, rel, 3) == 1.0


def test_empty_qrel_fails_fast():
    with pytest.raises(ValueError):
        m.hit_at_k(["A"], set(), 1)
    with pytest.raises(ValueError):
        m.compute_all(["A"], set())


def test_ndcg_idcg_independent_of_retrieved():
    # 2 relevant docs, only 1 retrieved at rank 1:
    # DCG = 1/log2(2) = 1.0 ; IDCG = 1/log2(2) + 1/log2(3)
    run, rel = ["A", "X", "Y"], {"A", "B"}
    expected = 1.0 / (1.0 + 1.0 / math.log2(3))
    assert m.ndcg_at_k(run, rel, 10) == pytest.approx(expected)


def test_set_em_appendix_only():
    assert m.set_em(["A", "B", "X"], {"A", "B"}) == 1.0
    assert m.set_em(["A", "X", "B"], {"A", "B"}) == 0.0


def test_mrr_zero_when_absent_in_topk():
    run = ["X1", "X2", "A"]
    assert m.mrr_at_k(run, {"A"}, 2) == 0.0
    assert m.mrr_at_k(run, {"A"}, 3) == pytest.approx(1 / 3)


# ── Cross-check vs ranx / pytrec_eval (1e-9 agreement) ────────────────────────

CASES = [
    (["A", "B", "C"], {"A"}),
    (["X", "Y", "A"], {"A"}),
    (["A", "X", "B"], {"A", "B"}),
    (["X", "Y", "Z"], {"A"}),                       # total miss
    (["B", "A", "C", "D", "E"], {"A", "C", "E"}),   # scattered relevants
]


def _to_ranx(cases):
    qrels = {f"q{i}": {d: 1 for d in rel} for i, (_, rel) in enumerate(cases)}
    run = {f"q{i}": {d: float(len(r) - j) for j, d in enumerate(r)}
           for i, (r, _) in enumerate(cases)}
    return qrels, run


def test_cross_check_ranx():
    ranx = pytest.importorskip("ranx")
    qrels_d, run_d = _to_ranx(CASES)
    qrels, run = ranx.Qrels(qrels_d), ranx.Run(run_d)
    for k in (1, 3, 5):
        got = ranx.evaluate(qrels, run, [f"hit_rate@{k}", f"recall@{k}",
                                         f"precision@{k}", f"mrr@{k}",
                                         f"map@{k}", f"ndcg@{k}"])
        ours = {f"hit_rate@{k}": 0.0, f"recall@{k}": 0.0, f"precision@{k}": 0.0,
                f"mrr@{k}": 0.0, f"map@{k}": 0.0, f"ndcg@{k}": 0.0}
        for r, rel in CASES:
            ours[f"hit_rate@{k}"] += m.hit_at_k(r, rel, k)
            ours[f"recall@{k}"] += m.recall_at_k(r, rel, k)
            ours[f"precision@{k}"] += m.precision_at_k(r, rel, k)
            ours[f"mrr@{k}"] += m.mrr_at_k(r, rel, k)
            ours[f"map@{k}"] += m.average_precision_at_k(r, rel, k)
            ours[f"ndcg@{k}"] += m.ndcg_at_k(r, rel, k)
        for key in ours:
            ours[key] /= len(CASES)
            assert abs(ours[key] - got[key]) < 1e-9, (
                f"{key}: ours={ours[key]} ranx={got[key]}")


def test_cross_check_pytrec_eval():
    pytrec_eval = pytest.importorskip("pytrec_eval")
    qrels_d, run_d = _to_ranx(CASES)
    evaluator = pytrec_eval.RelevanceEvaluator(
        qrels_d, {"recip_rank", "map_cut.10", "ndcg_cut.10", "recall.5"})
    got = evaluator.evaluate(run_d)
    for i, (r, rel) in enumerate(CASES):
        qid = f"q{i}"
        assert abs(got[qid]["recip_rank"] - m.mrr_at_k(r, rel, 1000)) < 1e-9
        assert abs(got[qid]["map_cut_10"]
                   - m.average_precision_at_k(r, rel, 10)) < 1e-9
        assert abs(got[qid]["ndcg_cut_10"] - m.ndcg_at_k(r, rel, 10)) < 1e-9
        assert abs(got[qid]["recall_5"] - m.recall_at_k(r, rel, 5)) < 1e-9
