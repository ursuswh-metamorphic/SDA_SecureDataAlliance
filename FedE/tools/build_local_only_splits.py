"""Build deterministic train/query splits for the B2 local-only baseline.

The company-to-client routing is derived from the training set with the same
LPT rule as ``CompanyPartitioner``.  Every company, training row, and query is
assigned exactly once.
"""
import argparse
import collections
import hashlib
import json
import os
import re


def normalize_company(value):
    company = re.sub(r"[^A-Z0-9]", "", str(value).strip().upper())
    if not company:
        raise ValueError("empty company label encountered")
    return company


def build_routing(rows, num_clients):
    if num_clients <= 0:
        raise ValueError("num_clients must be positive")
    counts = collections.Counter(normalize_company(row["company"]) for row in rows)
    if len(counts) < num_clients:
        raise ValueError(
            f"{len(counts)} companies cannot populate {num_clients} clients"
        )

    client_sizes = [0] * num_clients
    client_companies = [[] for _ in range(num_clients)]
    company_to_client = {}
    for company, count in sorted(counts.items(), key=lambda item: (-item[1], item[0])):
        client_id = min(range(num_clients), key=lambda cid: (client_sizes[cid], cid))
        company_to_client[company] = client_id
        client_companies[client_id].append(company)
        client_sizes[client_id] += count
    return company_to_client, client_companies, client_sizes


def split_rows(rows, company_to_client, company_field):
    splits = [[] for _ in range(max(company_to_client.values()) + 1)]
    for row in rows:
        company = normalize_company(row[company_field])
        if company not in company_to_client:
            raise ValueError(f"company {company!r} has no client route")
        splits[company_to_client[company]].append(row)
    return splits


def sha256_file(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_jsonl(path):
    with open(path, "r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train", required=True)
    parser.add_argument("--queries", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--num-clients", type=int, default=5)
    args = parser.parse_args()

    with open(args.train, "r", encoding="utf-8") as handle:
        train_rows = json.load(handle)
    query_rows = load_jsonl(args.queries)
    if not train_rows or not query_rows:
        raise ValueError("train and query inputs must both be non-empty")

    routing, client_companies, client_sizes = build_routing(
        train_rows, args.num_clients
    )
    train_splits = split_rows(train_rows, routing, "company")
    query_field = (
        "company_canonical"
        if all("company_canonical" in row for row in query_rows)
        else "company"
    )
    query_splits = split_rows(query_rows, routing, query_field)

    os.makedirs(args.output, exist_ok=True)
    clients = []
    for client_id, (train_split, query_split) in enumerate(
        zip(train_splits, query_splits)
    ):
        if not train_split or not query_split:
            raise ValueError(
                f"client {client_id} has train={len(train_split)}, "
                f"queries={len(query_split)}; B2 requires both"
            )
        train_path = os.path.join(args.output, f"client_{client_id}_train.json")
        query_path = os.path.join(args.output, f"client_{client_id}_queries.jsonl")
        with open(train_path, "w", encoding="utf-8", newline="\n") as handle:
            json.dump(train_split, handle, ensure_ascii=False, indent=2)
            handle.write("\n")
        with open(query_path, "w", encoding="utf-8", newline="\n") as handle:
            for row in query_split:
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")
        clients.append(
            {
                "client_id": client_id,
                "companies": client_companies[client_id],
                "n_train": len(train_split),
                "n_queries": len(query_split),
                "train_file": os.path.basename(train_path),
                "queries_file": os.path.basename(query_path),
            }
        )

    if sum(row["n_train"] for row in clients) != len(train_rows):
        raise AssertionError("training split is not exhaustive")
    if sum(row["n_queries"] for row in clients) != len(query_rows):
        raise AssertionError("query split is not exhaustive")

    manifest = {
        "protocol": "B2 local-only, company-routed",
        "num_clients": args.num_clients,
        "train_source": {
            "path": os.path.abspath(args.train),
            "sha256": sha256_file(args.train),
            "n_rows": len(train_rows),
        },
        "query_source": {
            "path": os.path.abspath(args.queries),
            "sha256": sha256_file(args.queries),
            "n_rows": len(query_rows),
        },
        "company_to_client": dict(sorted(routing.items())),
        "client_sizes_before_holdout": client_sizes,
        "clients": clients,
    }
    with open(
        os.path.join(args.output, "manifest.json"),
        "w",
        encoding="utf-8",
        newline="\n",
    ) as handle:
        json.dump(manifest, handle, ensure_ascii=False, indent=2)
        handle.write("\n")
    print(json.dumps(clients, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
