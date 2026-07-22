"""eval_clean.py — P3 clean-protocol evaluator (Giai đoạn B1).

Encodes the FROZEN corpus manifest, retrieves top-k per query, dedupes,
computes standard IR metrics (tools/ir_metrics.py), and writes:
  * TREC run file            <out>/run.trec
  * per-query metrics JSON   <out>/per_query.json
  * aggregate + manifest     <out>/result.json

Index independence (HARD gate in tools/audit_protocol.py): this file reads
ONLY corpus_v1.jsonl / queries_*.jsonl / qrels_*.tsv produced by the build
tools. It never opens raw QA annotation files and never adds documents at
eval time.

Fail-fast: every query must have at least one qrel and every qrel passage
must exist in the corpus — otherwise the run aborts before retrieval.

Primary endpoint (pre-declared): MRR@10. Everything else is secondary.

Usage:
  python -X utf8 eval_clean.py \
      --corpus ../artifacts/data/corpus_v1.jsonl \
      --queries ../artifacts/data/queries_val_v1.jsonl \
      --qrels ../artifacts/data/qrels_val_v1.tsv \
      --model BAAI/bge-base-en-v1.5 --top-k 100 \
      --output ../artifacts/eval/p3_pretrained_val

Smoke mode (pipeline check only, never for tables):
  ... --smoke-passages 500     # index = all qrel passages + 500 fillers
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from tools import ir_metrics  # noqa: E402


def sha256_file(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def load_corpus(path: str) -> tuple[list[str], list[str], dict]:
    """Return (passage_ids, texts, header) from corpus_v1.jsonl."""
    pids, texts, header = [], [], {}
    with open(path, encoding="utf-8") as f:
        for line in f:
            row = json.loads(line)
            if row.get("_header"):
                header = row
                continue
            pids.append(row["passage_id"])
            texts.append(row["text"])
    return pids, texts, header


def load_queries(path: str) -> list[dict]:
    out = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            out.append(json.loads(line))
    return out


def load_qrels(path: str) -> dict[str, set[str]]:
    qrels: dict[str, set[str]] = {}
    with open(path, encoding="utf-8") as f:
        for line in f:
            parts = line.split()
            if len(parts) != 4:
                continue
            qid, _, pid, rel = parts
            if int(rel) > 0:
                qrels.setdefault(qid, set()).add(pid)
    return qrels


def build_model(model_name: str, checkpoint: str | None, lora_checkpoint: str | None,
                device: str):
    from transformers import AutoModel, AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name)
    if checkpoint:
        state = torch.load(checkpoint, map_location="cpu", weights_only=True)
        state = {k.removeprefix("model."): v for k, v in state.items()}
        missing, unexpected = model.load_state_dict(state, strict=False)
        print(f"[eval_clean] loaded full checkpoint {checkpoint} "
              f"(missing={len(missing)}, unexpected={len(unexpected)})")
    if lora_checkpoint:
        try:
            from peft import PeftModel  # noqa: F401
        except ImportError as e:
            raise RuntimeError(
                "LoRA checkpoint evaluation requires peft; install it or merge "
                "the adapter into a full state dict first."
            ) from e
        raise NotImplementedError(
            "LoRA adapter loading is wired in Giai đoạn C (run_experiment.py "
            "merges adapters before eval)."
        )
    model.eval().to(device)
    return model, tokenizer


@torch.no_grad()
def encode(model, tokenizer, texts: list[str], device: str, batch_size: int,
           max_length: int, pooling: str, instruction: str = "") -> torch.Tensor:
    embs = []
    for i in range(0, len(texts), batch_size):
        batch = [instruction + t for t in texts[i:i + batch_size]]
        inp = tokenizer(batch, return_tensors="pt", padding=True,
                        truncation=True, max_length=max_length).to(device)
        out = model(**inp).last_hidden_state            # (B, T, D)
        if pooling == "cls":
            pooled = out[:, 0]
        elif pooling == "masked_mean":
            mask = inp["attention_mask"].unsqueeze(-1).float()
            pooled = (out * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1e-9)
        else:
            raise ValueError(f"unknown pooling: {pooling}")
        pooled = torch.nn.functional.normalize(pooled, p=2, dim=-1)
        embs.append(pooled.cpu())
        if (i // batch_size) % 50 == 0:
            print(f"  encoded {i + len(batch)}/{len(texts)}")
    return torch.cat(embs, dim=0)


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--corpus", required=True)
    p.add_argument("--queries", required=True)
    p.add_argument("--qrels", required=True)
    p.add_argument("--model", default="BAAI/bge-base-en-v1.5")
    p.add_argument("--checkpoint", default=None, help="full state_dict .bin")
    p.add_argument("--lora-checkpoint", default=None)
    p.add_argument("--top-k", type=int, default=100)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--max-length", type=int, default=512)
    p.add_argument("--pooling", choices=["cls", "masked_mean"], default="cls",
                   help="cls = BGE model-card pooling (default)")
    p.add_argument("--query-instruction", default="",
                   help="optional BGE query prefix; recorded in the manifest")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--seed", type=int, default=13)
    p.add_argument("--run-name", default=None)
    p.add_argument("--output", required=True)
    p.add_argument("--smoke-passages", type=int, default=0,
                   help="SMOKE ONLY: index = qrel passages + N fillers; output "
                        "is tagged smoke=true and must never enter a table")
    args = p.parse_args()

    torch.manual_seed(args.seed)

    pids, texts, header = load_corpus(args.corpus)
    queries = load_queries(args.queries)
    qrels = load_qrels(args.qrels)

    # ── Fail-fast validation (no silent skips) ────────────────────────────
    pid_set = set(pids)
    missing_qrel = [q["qid"] for q in queries if q["qid"] not in qrels]
    if missing_qrel:
        raise RuntimeError(f"{len(missing_qrel)} queries without qrels "
                           f"(e.g. {missing_qrel[:5]}) — fail-fast, no skipping.")
    dangling = {pid for rel in qrels.values() for pid in rel if pid not in pid_set}
    if dangling:
        raise RuntimeError(f"{len(dangling)} qrel passage_ids missing from corpus "
                           f"— corpus/qrels mismatch, aborting before retrieval.")

    smoke = args.smoke_passages > 0
    if smoke:
        keep = {pid for rel in qrels.values() for pid in rel}
        filler = [i for i, pid in enumerate(pids) if pid not in keep][:args.smoke_passages]
        keep_idx = sorted(set(filler) | {i for i, pid in enumerate(pids) if pid in keep})
        pids = [pids[i] for i in keep_idx]
        texts = [texts[i] for i in keep_idx]
        print(f"[eval_clean] SMOKE MODE: {len(pids)} passages "
              f"({len(keep)} qrel + {len(filler)} filler) — NOT a real run.")

    model, tokenizer = build_model(args.model, args.checkpoint,
                                   args.lora_checkpoint, args.device)

    print(f"[eval_clean] encoding {len(texts)} passages...")
    t0 = time.time()
    p_embs = encode(model, tokenizer, texts, args.device, args.batch_size,
                    args.max_length, args.pooling)
    t_corpus = time.time() - t0

    print(f"[eval_clean] encoding {len(queries)} queries...")
    t0 = time.time()
    q_texts = [q["question"] for q in queries]
    q_embs = encode(model, tokenizer, q_texts, args.device, args.batch_size,
                    args.max_length, args.pooling,
                    instruction=args.query_instruction)
    t_query = time.time() - t0

    sims = q_embs @ p_embs.t()
    k = min(args.top_k, len(pids))
    scores, indices = torch.topk(sims, k=k, dim=-1)

    run_name = args.run_name or os.path.basename(args.output.rstrip("/\\"))
    os.makedirs(args.output, exist_ok=True)

    per_query = []
    trec_lines = []
    for qi, q in enumerate(queries):
        retrieved = [pids[int(i)] for i in indices[qi].tolist()]
        retrieved = ir_metrics.dedupe(retrieved)
        rel = qrels[q["qid"]]
        m = ir_metrics.compute_all(retrieved, rel)
        per_query.append({
            "qid": q["qid"],
            "company": q.get("company_canonical", q.get("company", "?")),
            "n_qrels": len(rel),
            "metrics": m,
            "retrieved_top10": retrieved[:10],
        })
        for rank, (pid, sc) in enumerate(zip(retrieved,
                                             scores[qi].tolist()), start=1):
            trec_lines.append(f"{q['qid']} Q0 {pid} {rank} {sc:.6f} {run_name}")

    n = len(per_query)
    agg = {}
    for key in per_query[0]["metrics"]:
        agg[key] = round(sum(r["metrics"][key] for r in per_query) / n * 100, 2)

    with open(os.path.join(args.output, "run.trec"), "w", encoding="utf-8",
              newline="\n") as f:
        f.write("\n".join(trec_lines) + "\n")
    with open(os.path.join(args.output, "per_query.json"), "w",
              encoding="utf-8") as f:
        json.dump(per_query, f, indent=2, ensure_ascii=False)

    result = {
        "run_name": run_name,
        "smoke": smoke,
        "primary_endpoint": ir_metrics.PRIMARY_METRIC,
        "primary_value": agg[ir_metrics.PRIMARY_METRIC],
        "aggregate": agg,
        "n_queries": n,
        "n_passages_indexed": len(pids),
        "manifest": {
            "corpus": {"path": args.corpus, "sha256": sha256_file(args.corpus),
                       "normalize_version": header.get("normalize_version")},
            "queries": {"path": args.queries, "sha256": sha256_file(args.queries)},
            "qrels": {"path": args.qrels, "sha256": sha256_file(args.qrels)},
            "model": args.model,
            "checkpoint": args.checkpoint,
            "pooling": args.pooling,
            "query_instruction": args.query_instruction,
            "top_k": args.top_k,
            "max_length": args.max_length,
            "seed": args.seed,
            "device": args.device,
            "torch": torch.__version__,
            "encode_seconds": {"corpus": round(t_corpus, 1),
                               "queries": round(t_query, 1)},
        },
    }
    with open(os.path.join(args.output, "result.json"), "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)

    tag = " [SMOKE — not for tables]" if smoke else ""
    print(f"\n=== eval_clean: {run_name}{tag} (n={n}) ===")
    print(f"  PRIMARY {ir_metrics.PRIMARY_METRIC} = {agg[ir_metrics.PRIMARY_METRIC]:.2f}")
    print(f"  hit@1/5/10 = {agg['hit@1']:.2f}/{agg['hit@5']:.2f}/{agg['hit@10']:.2f}  "
          f"recall@10 = {agg['recall@10']:.2f}  ndcg@10 = {agg['ndcg@10']:.2f}")
    print(f"  outputs -> {args.output}")


if __name__ == "__main__":
    main()
