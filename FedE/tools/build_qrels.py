"""build_qrels.py — map QA annotations to canonical passage IDs (Giai đoạn A2).

Primary route (A2a): every golden reference was restored into corpus_v1.jsonl
by build_corpus_manifest.py with original_index = reference_idx, so resolution
is a direct original_index -> passage_id lookup (coverage 100% by
construction). Duplicate-dropped restored pages resolve through
<corpus>.dropped_duplicates.jsonl (duplicate_of pointer).

The old mapping chain — (doc_name, evidence_page_num) direct match — is kept
ONLY as a cross-check column (`xcheck`), never as the resolution mechanism
(measured ceiling ~30%, and even "matching" pages carry different content
because evidence pagination differs from corpus pagination).

Outputs:
  qrels TSV (TREC):    qid 0 passage_id 1
  queries JSONL:       {"qid", "question", "company", "company_canonical",
                        "doc_name", "n_qrels"}
  unresolved TSV:      qid, reference_idx, doc_hint, page_hint, issue

Usage:
  python -X utf8 tools/build_qrels.py --qa paper_test_data/val_qa_data_50.json \
      --corpus ../artifacts/data/corpus_v1.jsonl --split val \
      --output ../artifacts/data/qrels_val_v1.tsv \
      --queries-out ../artifacts/data/queries_val_v1.jsonl \
      --unresolved ../artifacts/data/unresolved_val.tsv
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from tools.normalize import norm_doc_name  # noqa: E402

_ALNUM_RE = re.compile(r"[^A-Z0-9]")


def load_company_map(path: str | None) -> dict:
    aliases = {}
    if path and os.path.exists(path):
        with open(path, encoding="utf-8") as f:
            aliases = json.load(f).get("aliases", {})
    return aliases


def canonical_company(name: str, aliases: dict) -> str:
    c = _ALNUM_RE.sub("", str(name).upper())
    return aliases.get(c, c)


def load_corpus_maps(corpus_path: str):
    """Return (idx2pids, docpage2pid, pid_set) from corpus_v1.jsonl (+dropped).

    idx2pids maps original_index -> SET of passage_ids: the same golden page
    can be excerpted differently by val and test QA (12 such cases measured),
    producing two restored passages that share original_index. Both are
    legitimate golden variants, so qrels emit every one of them.
    """
    idx2pids: dict[int, set[str]] = {}
    docpage2pid: dict[tuple[str, int], str] = {}
    pid_set: set[str] = set()
    with open(corpus_path, encoding="utf-8") as f:
        for line in f:
            row = json.loads(line)
            if row.get("_header"):
                continue
            pid = row["passage_id"]
            pid_set.add(pid)
            if row.get("original_index") is not None:
                idx2pids.setdefault(int(row["original_index"]), set()).add(pid)
            docpage2pid[(norm_doc_name(row["doc_name"]), int(row["page_num"]))] = pid
    dropped_path = corpus_path + ".dropped_duplicates.jsonl"
    if os.path.exists(dropped_path):
        with open(dropped_path, encoding="utf-8") as f:
            for line in f:
                row = json.loads(line)
                oi = row.get("original_index")
                if oi is not None:
                    idx2pids.setdefault(int(oi), set()).add(row["duplicate_of"])
    return idx2pids, docpage2pid, pid_set


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--qa", required=True)
    p.add_argument("--corpus", required=True, help="corpus_v1.jsonl")
    p.add_argument("--split", required=True, choices=["val", "test"])
    p.add_argument("--output", required=True, help="qrels TSV (TREC format)")
    p.add_argument("--queries-out", default=None)
    p.add_argument("--unresolved", required=True)
    p.add_argument("--company-map",
                   default=os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                        "company_map.json"))
    args = p.parse_args()

    aliases = load_company_map(args.company_map)
    idx2pids, docpage2pid, pid_set = load_corpus_maps(args.corpus)

    with open(args.qa, encoding="utf-8") as f:
        qa = json.load(f)

    qrel_rows: list[tuple[str, str, str]] = []   # (qid, pid, xcheck)
    query_rows: list[dict] = []
    unresolved: list[tuple] = []
    xcheck_stats = {"xcheck_page_ok": 0, "xcheck_page_mismatch": 0, "xcheck_page_absent": 0}

    for pos, entry in enumerate(qa):
        qid = f"{args.split}_{pos:04d}"
        kc = entry.get("key_content", {})
        oi = entry.get("other_info", {})
        refs = kc.get("reference", [])
        ref_ids = kc.get("reference_idx", [])
        evs = oi.get("evidence", [])
        n_resolved = 0

        aligned = len(refs) == len(ref_ids) == len(evs)
        if not aligned:
            unresolved.append((qid, "", oi.get("doc_name", ""), "",
                               f"misaligned reference/reference_idx/evidence "
                               f"({len(refs)}/{len(ref_ids)}/{len(evs)})"))

        pairs = zip(ref_ids, evs) if aligned else [(r, {}) for r in ref_ids]
        for ref_id, ev in pairs:
            pids = idx2pids.get(int(ref_id), set()) if ref_id is not None else set()
            pids = {p for p in pids if p in pid_set}
            if not pids:
                unresolved.append((qid, ref_id, ev.get("doc_name", ""),
                                   ev.get("evidence_page_num", ""),
                                   "reference_idx not resolvable in corpus"))
                continue
            # Cross-check route (never decides resolution).
            doc, pn = ev.get("doc_name"), ev.get("evidence_page_num")
            if isinstance(pn, list):
                pn = pn[0] if pn else None
            alt = None
            if doc is not None and pn is not None:
                alt = docpage2pid.get((norm_doc_name(doc), int(pn)))
            for pid in sorted(pids):
                if alt is None:
                    xcheck = "xcheck_page_absent"
                elif alt == pid:
                    xcheck = "xcheck_page_ok"
                else:
                    xcheck = "xcheck_page_mismatch"
                xcheck_stats[xcheck] += 1
                qrel_rows.append((qid, pid, xcheck))
            n_resolved += 1

        query_rows.append({
            "qid": qid,
            "question": kc.get("question", ""),
            "company": oi.get("company", ""),
            "company_canonical": canonical_company(oi.get("company", ""), aliases),
            "doc_name": oi.get("doc_name", ""),
            "n_qrels": n_resolved,
        })

    # Dedupe (qid, pid) pairs, keep first xcheck.
    seen = set()
    deduped = []
    for qid, pid, xcheck in qrel_rows:
        if (qid, pid) in seen:
            continue
        seen.add((qid, pid))
        deduped.append((qid, pid, xcheck))

    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    with open(args.output, "w", encoding="utf-8", newline="\n") as f:
        for qid, pid, _ in deduped:
            f.write(f"{qid} 0 {pid} 1\n")
    # Sidecar with match_method / xcheck detail (not TREC, for audit).
    with open(args.output + ".detail.tsv", "w", encoding="utf-8", newline="\n") as f:
        f.write("qid\tpassage_id\tmatch_method\txcheck\n")
        for qid, pid, xcheck in deduped:
            f.write(f"{qid}\t{pid}\trestored_original_index\t{xcheck}\n")

    if args.queries_out:
        with open(args.queries_out, "w", encoding="utf-8", newline="\n") as f:
            for row in query_rows:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")

    with open(args.unresolved, "w", encoding="utf-8", newline="\n") as f:
        f.write("qid\treference_idx\tdoc_hint\tpage_hint\tissue\n")
        for row in unresolved:
            f.write("\t".join(str(x) for x in row) + "\n")

    n_q = len(query_rows)
    n_cov = sum(1 for r in query_rows if r["n_qrels"] > 0)
    print(f"[build_qrels] split={args.split}: {n_q} queries, {len(deduped)} qrels, "
          f"coverage {n_cov}/{n_q}, unresolved rows {len(unresolved)}")
    print(f"[build_qrels] cross-check: {xcheck_stats}")
    if n_cov < n_q:
        print(f"[build_qrels] WARNING: {n_q - n_cov} queries have 0 qrels "
              f"-> audit_protocol will fail-fast.")


if __name__ == "__main__":
    main()
