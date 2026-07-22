"""build_corpus_manifest.py — P3 clean corpus with canonical passage IDs (Giai đoạn A1+A2a).

Responsibilities (and ONLY these):
  1. test_corpus.json → corpus_v1.jsonl, one passage per page, with stable
     content-derived passage_id (see tools/normalize.py). No incremental IDs.
  2. Restore-mode (A2a): the released dataset physically excised exactly the
     168 golden evidence pages from BOTH train_corpus and test_corpus (their
     texts survive only in QA key_content.reference). With --restore-from-qa
     those pages are re-added ONCE, at corpus-build time, BEFORE freeze, with
     full provenance (doc_name, evidence_page_num, original reference_idx,
     source_split='qa_reference_restored', restored=true).

     This is NOT eval-time oracle injection (P0.1): it happens once at build
     time, is fully disclosed in the manifest header, and the evaluator
     (eval_clean.py) never opens a QA file.
  3. Exact dedupe by normalized-text SHA-256 (first occurrence wins; corpus
     pages are processed before restored pages, so a restored page whose text
     already exists resolves to the existing passage). Dropped rows go to
     <output>.dropped_duplicates.jsonl with a duplicate_of pointer so
     build_qrels can still resolve them.

Output JSONL row schema:
  {"passage_id", "parent_document_id", "doc_name", "page_num", "text",
   "text_sha256", "source_split", "original_index", "restored"}

First line of the output file is a header record:
  {"_header": true, "normalize_version", "build_args", "counts", ...}

Usage:
  python -X utf8 tools/build_corpus_manifest.py \
      --input paper_test_data/test_corpus.json \
      --restore-from-qa paper_test_data/val_qa_data_50.json paper_test_data/test_qa_data_100.json \
      --restored-out ../artifacts/data/restored_pages_v1.jsonl \
      --output ../artifacts/data/corpus_v1.jsonl
"""
from __future__ import annotations

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from tools.normalize import (  # noqa: E402
    NORMALIZE_VERSION,
    norm_text,
    parent_document_id,
    passage_id,
    text_fingerprint,
)


def iter_corpus_pages(corpus_path: str):
    """Yield (doc_name, page_num:int, text, original_index) for every
    non-empty page of the nested corpus dict."""
    with open(corpus_path, encoding="utf-8") as f:
        corpus = json.load(f)
    for doc_name, pages in corpus.items():
        for page_num_str, page_data in pages.items():
            if not isinstance(page_data, dict):
                continue
            text = page_data.get("page_content", "")
            if not text or not norm_text(text):
                continue
            yield doc_name, int(page_num_str), text, page_data.get("index")


def iter_restore_pages(qa_paths: list[str]):
    """Yield (doc_name, page_num, text, reference_idx, qa_file, entry_pos, issue)
    for every (reference, reference_idx, evidence) triple in the QA files.

    reference[i] / reference_idx[i] / evidence[i] are aligned by position.
    Misaligned entries yield issue != None and NO page (they go to the
    restore-issues report; build_qrels marks the query unresolved).
    """
    for qa_path in qa_paths:
        with open(qa_path, encoding="utf-8") as f:
            qa = json.load(f)
        for pos, entry in enumerate(qa):
            kc = entry.get("key_content", {})
            refs = kc.get("reference", [])
            ref_ids = kc.get("reference_idx", [])
            evs = entry.get("other_info", {}).get("evidence", [])
            if not (len(refs) == len(ref_ids) == len(evs)):
                yield (None, None, None, None, qa_path, pos,
                       f"misaligned lists: reference={len(refs)} "
                       f"reference_idx={len(ref_ids)} evidence={len(evs)}")
                continue
            for ref_text, ref_id, ev in zip(refs, ref_ids, evs):
                if not ref_text or not norm_text(ref_text):
                    yield (None, None, None, ref_id, qa_path, pos, "empty reference text")
                    continue
                doc = ev.get("doc_name")
                pn = ev.get("evidence_page_num")
                if isinstance(pn, list):
                    pn = pn[0] if pn else None
                if doc is None or pn is None:
                    yield (None, None, None, ref_id, qa_path, pos, "missing doc/page in evidence")
                    continue
                yield doc, int(pn), ref_text, ref_id, qa_path, pos, None


def build(args: argparse.Namespace) -> dict:
    rows: list[dict] = []
    dropped: list[dict] = []
    restore_issues: list[dict] = []
    seen_fp: dict[str, str] = {}          # text_sha256 -> passage_id (first wins)
    seen_pid: dict[str, str] = {}         # passage_id -> text_sha256 (collision guard)

    def add_row(doc_name, page_num, text, source_split, original_index, restored):
        fp = text_fingerprint(text)
        pid = passage_id(doc_name, page_num, text)
        if fp in seen_fp:
            dropped.append({
                "passage_id": pid, "doc_name": doc_name, "page_num": page_num,
                "source_split": source_split, "original_index": original_index,
                "restored": restored, "text_sha256": fp,
                "duplicate_of": seen_fp[fp],
            })
            return
        if pid in seen_pid:
            raise RuntimeError(
                f"passage_id collision: {pid} (doc={doc_name!r} page={page_num}) "
                f"— same (doc,page,text) appeared twice with different fingerprints?"
            )
        seen_fp[fp] = pid
        seen_pid[pid] = fp
        rows.append({
            "passage_id": pid,
            "parent_document_id": parent_document_id(doc_name),
            "doc_name": doc_name,
            "page_num": page_num,
            "text": text,                      # raw text, unmodified
            "text_sha256": fp,
            "source_split": source_split,
            "original_index": original_index,
            "restored": restored,
        })

    # 1. Corpus pages FIRST (so restored duplicates resolve to corpus pages).
    n_corpus = 0
    for doc_name, page_num, text, orig_idx in iter_corpus_pages(args.input):
        add_row(doc_name, page_num, text, "test_corpus", orig_idx, False)
        n_corpus += 1

    # 2. Restored golden pages (A2a).
    n_restored = 0
    restored_rows: list[dict] = []
    if args.restore_from_qa:
        for doc, pn, text, ref_id, qa_path, pos, issue in iter_restore_pages(args.restore_from_qa):
            if issue is not None:
                restore_issues.append({
                    "qa_file": os.path.basename(qa_path), "entry_pos": pos,
                    "reference_idx": ref_id, "issue": issue,
                })
                continue
            before = len(rows)
            add_row(doc, pn, text, "qa_reference_restored", ref_id, True)
            n_restored += 1
            if len(rows) > before:
                restored_rows.append(rows[-1])

    header = {
        "_header": True,
        "schema": "corpus_v1",
        "normalize_version": NORMALIZE_VERSION,
        "build_args": {
            "input": os.path.basename(args.input),
            "restore_from_qa": [os.path.basename(p) for p in (args.restore_from_qa or [])],
        },
        "counts": {
            "corpus_pages": n_corpus,
            "restore_candidates": n_restored,
            "restored_kept": len(restored_rows),
            "dropped_duplicates": len(dropped),
            "restore_issues": len(restore_issues),
            "total_passages": len(rows),
        },
    }

    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as f:
        f.write(json.dumps(header, ensure_ascii=False) + "\n")
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    dropped_path = args.output + ".dropped_duplicates.jsonl"
    with open(dropped_path, "w", encoding="utf-8") as f:
        for r in dropped:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    if args.restored_out:
        os.makedirs(os.path.dirname(os.path.abspath(args.restored_out)), exist_ok=True)
        with open(args.restored_out, "w", encoding="utf-8") as f:
            for r in restored_rows:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")

    issues_path = args.output + ".restore_issues.jsonl"
    with open(issues_path, "w", encoding="utf-8") as f:
        for r in restore_issues:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    print(f"[build_corpus_manifest] corpus pages kept : {n_corpus - sum(1 for d in dropped if not d['restored'])}")
    print(f"[build_corpus_manifest] restored kept     : {len(restored_rows)} / {n_restored} candidates")
    print(f"[build_corpus_manifest] dropped duplicates: {len(dropped)} -> {dropped_path}")
    print(f"[build_corpus_manifest] restore issues    : {len(restore_issues)} -> {issues_path}")
    print(f"[build_corpus_manifest] TOTAL passages    : {len(rows)} -> {args.output}")
    return header


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--input", required=True, help="test_corpus.json")
    p.add_argument("--restore-from-qa", nargs="*", default=None,
                   help="QA json files whose (reference, reference_idx, evidence) "
                        "triples are restored as first-class passages (A2a)")
    p.add_argument("--restored-out", default=None,
                   help="Where to write the restored-pages provenance JSONL")
    p.add_argument("--output", required=True, help="corpus_v1.jsonl output path")
    args = p.parse_args()
    build(args)


if __name__ == "__main__":
    main()
