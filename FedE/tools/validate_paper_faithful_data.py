"""Validate the exact FedE4RAG training release used by the paper."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path


EXPECTED_TRAIN_SHA256 = (
    "e10e402a2189948eb11759f3cec65901bfff6471c6577e3726c240827f1f176a"
)
EXPECTED_CORPUS_SHA256 = (
    "009a967f9472ec71c42497ac11aae12db82417e9c4279ccaafa689de2d75f165"
)
EXPECTED_COMPANY_COUNTS = {
    "AES": 4842,
    "BOEING": 5302,
    "ACTIVISIONBLIZZARD": 6328,
    "PG": 8382,
    "PEPSICO": 18804,
}
EXPECTED_FIELDS = {"company", "page", "index", "reference", "question"}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", type=Path, required=True)
    parser.add_argument("--train-corpus", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    train_hash = sha256(args.train)
    corpus_hash = sha256(args.train_corpus)
    records = json.loads(args.train.read_text(encoding="utf-8"))
    corpus = json.loads(args.train_corpus.read_text(encoding="utf-8"))

    field_errors = [
        index
        for index, record in enumerate(records)
        if set(record) != EXPECTED_FIELDS
    ]
    company_counts = dict(Counter(row["company"] for row in records))
    empty_pairs = sum(
        not str(row["question"]).strip() or not str(row["reference"]).strip()
        for row in records
    )
    corpus_pages = sum(len(pages) for pages in corpus.values())

    gates = {
        "train_sha256": train_hash == EXPECTED_TRAIN_SHA256,
        "train_records": len(records) == 43658,
        "company_counts": company_counts == EXPECTED_COMPANY_COUNTS,
        "record_schema": not field_errors,
        "nonempty_pairs": empty_pairs == 0,
        "train_corpus_sha256": corpus_hash == EXPECTED_CORPUS_SHA256,
        "train_corpus_documents": len(corpus) == 368,
        "train_corpus_pages": corpus_pages == 23123,
    }
    report = {
        "source": "DocAILab/FedE4RAG_Dataset",
        "revision": "398304846743f184d36f2c35a3db58fa9be70a9d",
        "released_path": "FEDE4FIN/train_data/data_50000_random.json",
        "paper": "arXiv:2504.19101v1",
        "train_path": str(args.train.resolve()),
        "train_sha256": train_hash,
        "train_records": len(records),
        "company_counts": company_counts,
        "empty_pairs": empty_pairs,
        "schema_error_count": len(field_errors),
        "train_corpus_path": str(args.train_corpus.resolve()),
        "train_corpus_sha256": corpus_hash,
        "train_corpus_documents": len(corpus),
        "train_corpus_pages": corpus_pages,
        "client_order": [
            "AES",
            "BOEING",
            "ACTIVISIONBLIZZARD",
            "PG",
            "PEPSICO",
        ],
        "gates": gates,
        "pass": all(gates.values()),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))
    if not report["pass"]:
        raise SystemExit("paper-faithful data validation failed")


if __name__ == "__main__":
    main()
