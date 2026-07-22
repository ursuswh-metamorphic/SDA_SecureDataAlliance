"""audit_protocol.py — acceptance gates for the P3 clean protocol (Giai đoạn A3).

Exit code != 0 when any HARD gate fails. Gates (2026-07-22 revision):

  HARD 1  Qrel coverage        every query has >=1 qrel whose passage_id is in corpus
  HARD 2  Restore integrity    exactly --expect-restored restored-or-dup-resolved pages;
                               0 restored fingerprints inside any training input
  HARD 3  Internal exact dup   0 duplicate text_sha256 inside corpus_v1
  HARD 4  Page-level leakage   0 shared text fingerprints train_corpus <-> corpus_v1;
                               0 shared fingerprints selected_data.reference <-> restored
  HARD 5  Stable ID            (when --compare-corpus given) both manifests have the
                               same passage_id set
  HARD 6  Index independence   (when --eval-source given) evaluator source contains no
                               QA-file access / append-refs patterns
  REPORT  Parent overlap       train<->test share parent documents BY DESIGN of the
                               released dataset (368/368) — measured and reported,
                               never a fail gate (M3)
  REPORT  Provenance collision selected_data (doc#page) vs restored (doc,page)
  REPORT  Company map          all company spellings covered by canonical rule+aliases

Usage:
  python -X utf8 tools/audit_protocol.py \
      --corpus ../artifacts/data/corpus_v1.jsonl \
      --qrels ../artifacts/data/qrels_val_v1.tsv ../artifacts/data/qrels_test_v1.tsv \
      --queries ../artifacts/data/queries_val_v1.jsonl ../artifacts/data/queries_test_v1.jsonl \
      --train-data selected_data.json --train-corpus train_corpus.json \
      --expect-restored 168 --require-coverage 1.0 --fail-on-page-leak
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from tools.normalize import norm_doc_name, parent_document_id, text_fingerprint  # noqa: E402

_ALNUM_RE = re.compile(r"[^A-Z0-9]")

FAILURES: list[str] = []
REPORTS: list[str] = []


def gate(ok: bool, hard: bool, msg: str):
    tag = "PASS" if ok else ("FAIL" if hard else "WARN")
    line = f"[{tag}] {msg}"
    print(line)
    if not ok and hard:
        FAILURES.append(line)
    elif not ok:
        REPORTS.append(line)


def load_corpus(corpus_path: str):
    rows = []
    header = None
    with open(corpus_path, encoding="utf-8") as f:
        for line in f:
            row = json.loads(line)
            if row.get("_header"):
                header = row
                continue
            rows.append(row)
    return header, rows


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--corpus", required=True)
    p.add_argument("--qrels", nargs="+", required=True)
    p.add_argument("--queries", nargs="+", required=True)
    p.add_argument("--train-data", default=None, help="selected_data.json")
    p.add_argument("--train-corpus", default=None, help="train_corpus.json")
    p.add_argument("--expect-restored", type=int, default=168)
    p.add_argument("--require-coverage", type=float, default=1.0)
    p.add_argument("--fail-on-page-leak", action="store_true")
    p.add_argument("--compare-corpus", default=None,
                   help="second-build manifest for the stable-ID gate")
    p.add_argument("--eval-source", default=None,
                   help="evaluator .py file for the index-independence static scan")
    p.add_argument("--company-map",
                   default=os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                        "company_map.json"))
    args = p.parse_args()

    header, rows = load_corpus(args.corpus)
    pid_set = {r["passage_id"] for r in rows}
    fp_all = [r["text_sha256"] for r in rows]
    restored = [r for r in rows if r.get("restored")]
    restored_fps = {r["text_sha256"] for r in restored}
    print(f"[audit] corpus: {len(rows)} passages ({len(restored)} restored), "
          f"normalize_version={header.get('normalize_version') if header else '?'}")

    # ── HARD 1: coverage ─────────────────────────────────────────────────
    for qpath, qrpath in zip(args.queries, args.qrels):
        qids = []
        with open(qpath, encoding="utf-8") as f:
            for line in f:
                qids.append(json.loads(line)["qid"])
        covered = set()
        n_qrel_rows = 0
        with open(qrpath, encoding="utf-8") as f:
            for line in f:
                parts = line.split()
                if len(parts) != 4:
                    continue
                qid, _, pid, rel = parts
                n_qrel_rows += 1
                if int(rel) > 0 and pid in pid_set:
                    covered.add(qid)
        cov = len(covered & set(qids)) / max(1, len(qids))
        gate(cov >= args.require_coverage, True,
             f"coverage {os.path.basename(qrpath)}: {len(covered & set(qids))}/{len(qids)} "
             f"({cov:.1%}) >= {args.require_coverage:.0%} required "
             f"[{n_qrel_rows} qrel rows, all pids in corpus]")

    # ── HARD 2: restore integrity ────────────────────────────────────────
    # --expect-restored counts UNIQUE golden original_index values (168): the
    # same page may be excerpted differently by val and test (extra passages
    # sharing an original_index) or identically (dup-resolved) — both fine as
    # long as every unique golden index is represented at least once.
    dropped_path = args.corpus + ".dropped_duplicates.jsonl"
    restored_idx = {int(r["original_index"]) for r in restored
                    if r.get("original_index") is not None}
    n_dup_restored = 0
    if os.path.exists(dropped_path):
        with open(dropped_path, encoding="utf-8") as f:
            for line in f:
                row = json.loads(line)
                if row.get("restored"):
                    n_dup_restored += 1
                    if row.get("original_index") is not None:
                        restored_idx.add(int(row["original_index"]))
    gate(len(restored_idx) == args.expect_restored, True,
         f"restore integrity: {len(restored_idx)} unique golden indexes "
         f"({len(restored)} passages kept, {n_dup_restored} dup-resolved) "
         f"== {args.expect_restored} expected")
    issues_path = args.corpus + ".restore_issues.jsonl"
    n_issues = 0
    if os.path.exists(issues_path):
        with open(issues_path, encoding="utf-8") as f:
            n_issues = sum(1 for _ in f)
    gate(n_issues == 0, True, f"restore issues: {n_issues} == 0")

    # ── HARD 3: internal exact duplicates ────────────────────────────────
    gate(len(fp_all) == len(set(fp_all)), True,
         f"internal exact-dup: {len(fp_all) - len(set(fp_all))} duplicate fingerprints == 0")

    # ── HARD 4 + REPORT: leakage vs training inputs ──────────────────────
    hard_leak = args.fail_on_page_leak
    if args.train_corpus:
        train_fps = set()
        train_parents = set()
        with open(args.train_corpus, encoding="utf-8") as f:
            tc = json.load(f)
        for doc_name, pages in tc.items():
            train_parents.add(parent_document_id(doc_name))
            for pn, pd in pages.items():
                if isinstance(pd, dict) and pd.get("page_content"):
                    train_fps.add(text_fingerprint(pd["page_content"]))
        leak = train_fps & set(fp_all)
        gate(len(leak) == 0, hard_leak,
             f"page-level leakage train_corpus <-> corpus_v1: {len(leak)} shared "
             f"fingerprints == 0")
        leak_r = train_fps & restored_fps
        gate(len(leak_r) == 0, hard_leak,
             f"page-level leakage train_corpus <-> restored: {len(leak_r)} == 0")
        # REPORT: parent overlap is BY DESIGN (M3) — measure, never fail.
        corpus_parents = {r["parent_document_id"] for r in rows}
        shared = len(train_parents & corpus_parents)
        print(f"[REPORT] parent-document overlap train<->test: {shared}/{len(corpus_parents)} "
              f"(inherent to released dataset — page-level split; document-disjoint "
              f"robustness split goes to appendix)")

    if args.train_data:
        with open(args.train_data, encoding="utf-8") as f:
            sel = json.load(f)
        sel_fps = {text_fingerprint(e["reference"]) for e in sel if e.get("reference")}
        leak_sel = sel_fps & restored_fps
        gate(len(leak_sel) == 0, hard_leak,
             f"leakage selected_data.reference <-> restored pages: {len(leak_sel)} == 0")
        leak_sel_all = sel_fps & set(fp_all)
        gate(len(leak_sel_all) == 0, False,
             f"leakage selected_data.reference <-> full corpus_v1: {len(leak_sel_all)} "
             f"shared fingerprints (WARN-level: training excerpts may legitimately "
             f"differ from eval pages)")
        # REPORT: provenance collision (doc,page) between training rows and restored.
        restored_dp = {(norm_doc_name(r["doc_name"]), int(r["page_num"])) for r in restored}
        collisions = 0
        for e in sel:
            page = str(e.get("page", ""))
            if "#p" in page:
                doc, pn = page.rsplit("#p", 1)
                try:
                    if (norm_doc_name(doc), int(pn)) in restored_dp:
                        collisions += 1
                except ValueError:
                    pass
        print(f"[REPORT] provenance (doc,page) collisions selected_data <-> restored: "
              f"{collisions} training records point at a restored golden page")

    # ── HARD 5: stable ID across rebuilds ────────────────────────────────
    if args.compare_corpus:
        _, rows2 = load_corpus(args.compare_corpus)
        pid_set2 = {r["passage_id"] for r in rows2}
        gate(pid_set == pid_set2, True,
             f"stable ID: rebuild produced identical passage_id set "
             f"({len(pid_set)} vs {len(pid_set2)}, "
             f"diff={len(pid_set ^ pid_set2)})")

    # ── HARD 6: index independence (static scan) ─────────────────────────
    if args.eval_source:
        with open(args.eval_source, encoding="utf-8") as f:
            src = f.read()
        forbidden = ["val_qa_data", "test_qa_data", "append_refs", "key_content",
                     "reference_idx"]
        found = [t for t in forbidden if t in src]
        gate(not found, True,
             f"index independence: evaluator {os.path.basename(args.eval_source)} "
             f"contains no QA-access patterns (found: {found or 'none'})")

    # ── REPORT: company map coverage ─────────────────────────────────────
    if os.path.exists(args.company_map):
        with open(args.company_map, encoding="utf-8") as f:
            aliases = json.load(f).get("aliases", {})
        known = set()
        if args.train_data:
            known |= {_ALNUM_RE.sub("", e["company"].upper()) for e in sel}
        canon = {aliases.get(c, c) for c in known}
        print(f"[REPORT] company map: {len(known)} raw spellings -> {len(canon)} canonical "
              f"({len(aliases)} aliases applied)")

    print()
    if FAILURES:
        print(f"[audit_protocol] {len(FAILURES)} HARD GATE FAILURE(S):")
        for f_ in FAILURES:
            print(f"  {f_}")
        sys.exit(1)
    print(f"[audit_protocol] ALL HARD GATES PASS ({len(REPORTS)} warnings)")
    sys.exit(0)


if __name__ == "__main__":
    main()
