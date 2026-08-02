import json

import pytest

from tools.build_local_only_splits import build_routing, split_rows
from tools.merge_routed_eval import aggregate, merge


def test_routing_is_deterministic_balanced_and_company_exclusive():
    rows = (
        [{"company": "A"}] * 5
        + [{"company": "B"}] * 4
        + [{"company": "C"}] * 3
        + [{"company": "D"}] * 2
    )
    routing, companies, sizes = build_routing(rows, 2)
    assert routing == {"A": 0, "B": 1, "C": 1, "D": 0}
    assert companies == [["A", "D"], ["B", "C"]]
    assert sizes == [7, 7]
    splits = split_rows(rows, routing, "company")
    assert [len(split) for split in splits] == [7, 7]


def test_merge_rejects_duplicate_or_missing_queries(tmp_path):
    dirs = []
    for client_id, qid in enumerate(["q0", "q0"]):
        eval_dir = tmp_path / f"client_{client_id}"
        eval_dir.mkdir()
        (eval_dir / "per_query.json").write_text(
            json.dumps([{"qid": qid, "metrics": {"mrr@10": 1.0}}]),
            encoding="utf-8",
        )
        (eval_dir / "result.json").write_text(
            json.dumps({"manifest": {"checkpoint": f"c{client_id}.bin"}}),
            encoding="utf-8",
        )
        (eval_dir / "run.trec").write_text("", encoding="utf-8")
        dirs.append(str(eval_dir))
    with pytest.raises(ValueError, match="multiple client outputs"):
        merge(dirs, ["q0", "q1"])


def test_aggregate_matches_eval_clean_macro_average():
    rows = [
        {"metrics": {"mrr@10": 1.0, "hit@1": 1}},
        {"metrics": {"mrr@10": 0.25, "hit@1": 0}},
    ]
    assert aggregate(rows) == {"mrr@10": 62.5, "hit@1": 50.0}
