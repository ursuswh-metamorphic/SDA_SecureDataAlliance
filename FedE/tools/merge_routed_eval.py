"""Merge disjoint per-client eval_clean outputs into one B2 result."""
import argparse
import hashlib
import json
import os


def sha256_file(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def aggregate(per_query):
    if not per_query:
        raise ValueError("cannot aggregate an empty query set")
    keys = tuple(per_query[0]["metrics"])
    for row in per_query:
        if tuple(row["metrics"]) != keys:
            raise ValueError("per-query metric keys differ across clients")
    return {
        key: round(
            sum(row["metrics"][key] for row in per_query) / len(per_query) * 100,
            2,
        )
        for key in keys
    }


def merge(eval_dirs, expected_qids):
    per_query = []
    trec_lines = []
    components = []
    seen = set()
    for client_id, eval_dir in enumerate(eval_dirs):
        per_query_path = os.path.join(eval_dir, "per_query.json")
        result_path = os.path.join(eval_dir, "result.json")
        trec_path = os.path.join(eval_dir, "run.trec")
        with open(per_query_path, "r", encoding="utf-8") as handle:
            rows = json.load(handle)
        with open(result_path, "r", encoding="utf-8") as handle:
            result = json.load(handle)
        for row in rows:
            qid = row["qid"]
            if qid in seen:
                raise ValueError(f"query {qid!r} occurs in multiple client outputs")
            seen.add(qid)
            row["routed_client"] = client_id
            per_query.append(row)
        with open(trec_path, "r", encoding="utf-8") as handle:
            trec_lines.extend(line.rstrip("\n") for line in handle if line.strip())
        components.append(
            {
                "client_id": client_id,
                "eval_dir": os.path.abspath(eval_dir),
                "n_queries": len(rows),
                "checkpoint": result["manifest"].get("checkpoint"),
                "result_sha256": sha256_file(result_path),
            }
        )

    expected = set(expected_qids)
    if seen != expected:
        raise ValueError(
            f"routed qids mismatch: missing={sorted(expected - seen)[:5]}, "
            f"unexpected={sorted(seen - expected)[:5]}"
        )
    per_query.sort(key=lambda row: row["qid"])
    return per_query, trec_lines, components


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--eval-dir", action="append", required=True)
    parser.add_argument("--queries", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--run-name", default="b2_local_only_company_routed")
    args = parser.parse_args()

    with open(args.queries, "r", encoding="utf-8") as handle:
        query_rows = [json.loads(line) for line in handle if line.strip()]
    expected_qids = [row["qid"] for row in query_rows]
    if len(expected_qids) != len(set(expected_qids)):
        raise ValueError("source queries contain duplicate qids")

    per_query, trec_lines, components = merge(args.eval_dir, expected_qids)
    metrics = aggregate(per_query)
    os.makedirs(args.output, exist_ok=True)
    with open(
        os.path.join(args.output, "per_query.json"), "w", encoding="utf-8"
    ) as handle:
        json.dump(per_query, handle, ensure_ascii=False, indent=2)
    with open(
        os.path.join(args.output, "run.trec"), "w", encoding="utf-8", newline="\n"
    ) as handle:
        handle.write("\n".join(trec_lines) + "\n")

    result = {
        "run_name": args.run_name,
        "smoke": False,
        "protocol": "B2 local-only, company-routed",
        "primary_endpoint": "mrr@10",
        "primary_value": metrics["mrr@10"],
        "aggregate": metrics,
        "n_queries": len(per_query),
        "query_source": {
            "path": os.path.abspath(args.queries),
            "sha256": sha256_file(args.queries),
        },
        "components": components,
    }
    with open(
        os.path.join(args.output, "result.json"),
        "w",
        encoding="utf-8",
        newline="\n",
    ) as handle:
        json.dump(result, handle, ensure_ascii=False, indent=2)
        handle.write("\n")
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
