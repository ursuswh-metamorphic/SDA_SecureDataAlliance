"""
eval_paper_faithful.py — paper-faithful retrieval evaluation for FedE4RAG.

Reproduces the upstream-evaluation methodology described in
"Privacy-Preserving Federated Embedding Learning for Localized RAG"
(arXiv:2504.19101, FedE4RAG), Tables II and III:

  * For each query in val_qa_data_50.json (or test_qa_data_100.json),
    encode the question with the model.
  * Encode every page of test_corpus.json (~30,829 pages) ONCE per model.
  * Rank all corpus pages by cosine similarity to the query.
  * A retrieved page is "relevant" iff its (doc_name, page_num) matches
    any of the golden pages from `record['other_info']['evidence']`.

Reports six metrics matching the paper:
  * Hit@1, Hit@10           (presence at specific rank cutoffs)
  * EM (Exact Match)        (any golden page in top-N retrieved)
  * MRR                     (reciprocal rank of first relevant page)
  * MAP (Mean Avg Precision)
  * NDCG                    (with binary relevance @ k=10)

Usage:
  python FedE/eval_paper_faithful.py                          # pretrained on val
  python FedE/eval_paper_faithful.py --checkpoint <ckpt.bin> --name dp_lora --split val
  python FedE/eval_paper_faithful.py --split test --smoke      # tiny synthetic check

CLI:
  --checkpoint   path to LoRA-only or full BertModel state_dict (default: pretrained)
  --name         display label (default: filename or 'pretrained')
  --split        val | test  (default: val)
  --corpus       path to corpus json (default: FedE/paper_test_data/test_corpus.json)
  --top-n        cutoff for retrieval (default: 100)
  --output-dir   where to dump JSON results (default: FedE/paper_test_data/eval_outputs/)
  --smoke        only first 5 queries × first 200 corpus pages (sanity check)
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from typing import Iterable, List, Tuple

import numpy as np
import torch
import torch.nn.functional as F


# ══════════════════════════════════════════════════════════════════════════════
#   Metrics — pure functions, no torch / model needed
# ══════════════════════════════════════════════════════════════════════════════

def hit_at_k(retrieved: List[Tuple[str, int]],
             golden: List[Tuple[str, int]],
             k: int) -> int:
    """1 if any golden page is in retrieved[:k], else 0."""
    if k <= 0 or not retrieved or not golden:
        return 0
    golden_set = set(golden)
    return int(any(p in golden_set for p in retrieved[:k]))


def em_at_n(retrieved: List[Tuple[str, int]],
            golden: List[Tuple[str, int]],
            n: int = 10) -> int:
    """Exact Match: any golden page found in retrieved[:n]. Same formula as Hit@n
    in the paper's presence-based metrics — separately reported because paper
    treats EM as binary "any hit" without ranking emphasis."""
    return hit_at_k(retrieved, golden, n)


def mrr_for_query(retrieved: List[Tuple[str, int]],
                  golden: List[Tuple[str, int]]) -> float:
    """Reciprocal rank of the FIRST relevant page in retrieved. 0 if none."""
    if not retrieved or not golden:
        return 0.0
    golden_set = set(golden)
    for rank, p in enumerate(retrieved, start=1):
        if p in golden_set:
            return 1.0 / rank
    return 0.0


def average_precision(retrieved: List[Tuple[str, int]],
                      golden: List[Tuple[str, int]]) -> float:
    """Average Precision for one query.

    Paper-faithful definition (standard IR):
      AP = sum( precision(k) for k where retrieved[k] in golden ) / |golden|
    where precision(k) = (count of relevants in retrieved[:k]) / k.

    Returns 0 if no relevant retrieved or golden empty.
    """
    if not retrieved or not golden:
        return 0.0
    golden_set = set(golden)
    hits = 0
    score = 0.0
    for k, p in enumerate(retrieved, start=1):
        if p in golden_set:
            hits += 1
            score += hits / k
    if hits == 0:
        return 0.0
    return score / len(golden_set)


def ndcg_at_k(retrieved: List[Tuple[str, int]],
              golden: List[Tuple[str, int]],
              k: int = 10) -> float:
    """NDCG@k with binary relevance.

    DCG@k = sum_{i=1..k} rel_i / log2(i+1)  where rel_i in {0,1}
    IDCG@k = DCG of the perfect ranking (min(k, |golden|) ones at the top)
    NDCG@k = DCG@k / IDCG@k    (0 if IDCG@k == 0)
    """
    if k <= 0 or not retrieved or not golden:
        return 0.0
    golden_set = set(golden)
    dcg = 0.0
    for i, p in enumerate(retrieved[:k], start=1):
        if p in golden_set:
            dcg += 1.0 / np.log2(i + 1)
    n_rel = min(k, len(golden_set))
    idcg = sum(1.0 / np.log2(i + 1) for i in range(1, n_rel + 1))
    return dcg / idcg if idcg > 0 else 0.0


# ══════════════════════════════════════════════════════════════════════════════
#   Data loading helpers
# ══════════════════════════════════════════════════════════════════════════════

def iter_corpus(corpus_dict: dict) -> Iterable[Tuple[str, int, str]]:
    """Yield (doc_name, page_num: int, page_text: str) tuples for every page
    in the nested corpus dict (doc_name -> {page_num: {page_content, ...}})."""
    for doc_name, pages in corpus_dict.items():
        for page_num_str, page_data in pages.items():
            text = page_data.get('page_content', '') if isinstance(page_data, dict) else str(page_data)
            yield doc_name, int(page_num_str), text


def golden_pages(record: dict) -> List[Tuple[str, int]]:
    """Extract list of (doc_name, page_num) golden pages from one query record."""
    out = []
    for ev in record.get('other_info', {}).get('evidence', []):
        doc = ev.get('doc_name')
        pn = ev.get('evidence_page_num')
        if doc is None or pn is None:
            continue
        # evidence_page_num is always int (verified in SCHEMA.md), but be defensive
        if isinstance(pn, list):
            for p in pn:
                out.append((doc, int(p)))
        else:
            out.append((doc, int(pn)))
    return out


def load_queries(split: str, smoke: bool = False) -> List[dict]:
    """Load queries from val or test split."""
    base = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'paper_test_data')
    fname = 'val_qa_data_50.json' if split == 'val' else 'test_qa_data_100.json'
    path = os.path.join(base, fname)
    with open(path, encoding='utf-8') as f:
        data = json.load(f)
    if smoke:
        data = data[:5]
    return data


# ══════════════════════════════════════════════════════════════════════════════
#   Embedding + retrieval
# ══════════════════════════════════════════════════════════════════════════════

def encode_texts(model, tokenizer, texts: List[str], device, batch_size: int = 32,
                 max_length: int | None = None) -> torch.Tensor:
    """Encode a list of strings into mean-pooled CLS-equivalent embeddings.

    Returns a (N, dim) torch tensor on CPU, L2-normalized.
    """
    if max_length is None:
        max_length = tokenizer.model_max_length
    embs = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i + batch_size]
        inp = tokenizer(batch, return_tensors='pt', padding=True,
                        truncation=True, max_length=max_length).to(device)
        with torch.no_grad():
            out = model(**inp)
        # Mean-pool over tokens (matches eval_phase2.py:46-50 + main_dp_lora_eps20.py:105)
        emb = out.last_hidden_state.mean(dim=1)
        embs.append(emb.cpu())
    e = torch.cat(embs, dim=0)
    return F.normalize(e, dim=-1, p=2)


def retrieve_top_n(query_embs: torch.Tensor, corpus_embs: torch.Tensor,
                   n: int = 100) -> Tuple[torch.Tensor, torch.Tensor]:
    """Return (indices, scores) of top-n corpus pages per query.
    query_embs: (Q, D) normalized
    corpus_embs: (C, D) normalized
    Returns indices (Q, n) and scores (Q, n), sorted descending.
    """
    sims = query_embs @ corpus_embs.t()         # (Q, C)
    n = min(n, corpus_embs.shape[0])
    scores, indices = torch.topk(sims, k=n, dim=-1, largest=True, sorted=True)
    return indices, scores


# ══════════════════════════════════════════════════════════════════════════════
#   Model loader (handles 3 checkpoint formats)
# ══════════════════════════════════════════════════════════════════════════════

def load_model(checkpoint_path: str | None):
    """Load BGE-base, optionally apply a LoRA-only or full state_dict.

    Three formats supported (matches eval_compare.py:201-222 logic):
      1. None / missing checkpoint -> pretrained zero-shot BGE-base.
      2. LoRA-only state_dict (keys contain 'lora') -> wrap with PEFT, merge.
      3. Full BertModel state_dict (with optional 'model.' prefix) -> load directly.
    """
    from transformers import BertModel
    base = BertModel.from_pretrained('BAAI/bge-base-en')

    if checkpoint_path is None or checkpoint_path == '' or not os.path.exists(checkpoint_path):
        if checkpoint_path:
            print(f'  [load_model] checkpoint not found: {checkpoint_path} -> using pretrained.')
        return base, 'pretrained_zero_shot'

    print(f'  [load_model] Loading checkpoint: {checkpoint_path}')
    state = torch.load(checkpoint_path, map_location='cpu', weights_only=True)

    has_lora = any('lora' in k.lower() for k in state.keys())
    if has_lora:
        # LoRA-only: wrap with PEFT, load, merge.
        from peft import LoraConfig, get_peft_model
        from flgo.benchmark.fedrag_classification.config import (
            LORA_R, LORA_ALPHA, LORA_TARGETS, LORA_DROPOUT,
        )
        cfg = LoraConfig(
            r=LORA_R, lora_alpha=LORA_ALPHA,
            target_modules=LORA_TARGETS,
            lora_dropout=LORA_DROPOUT, bias='none',
        )
        wrapped = get_peft_model(base, cfg)
        miss, unexp = wrapped.load_state_dict(state, strict=False)
        print(f'    LoRA-only ({len(state)} tensors); '
              f'{len(miss)} base keys missing (expected); {len(unexp)} unexpected.')
        if unexp:
            print(f'    [WARN] unexpected keys (sample): {list(unexp)[:3]}')
        merged = wrapped.merge_and_unload()
        return merged, 'lora_merged'

    # Full BertModel state_dict (legacy format).
    clean = {}
    for k, v in state.items():
        nk = k.replace('module.', '').replace('model.', '', 1) if 'model.' in k else k
        clean[nk] = v
    miss, unexp = base.load_state_dict(clean, strict=False)
    print(f'    Full-base ({len(state)} tensors); missing={len(miss)}, unexpected={len(unexp)}.')
    return base, 'full_base'


# ══════════════════════════════════════════════════════════════════════════════
#   Main eval loop
# ══════════════════════════════════════════════════════════════════════════════

def run_eval(checkpoint_path: str | None, name: str, split: str,
             corpus_path: str, top_n: int, smoke: bool,
             output_dir: str) -> dict:
    """Run one full eval. Returns the metrics dict; also writes JSON to disk."""
    from transformers import BertTokenizer

    print('=' * 70)
    print(f'  Setup: {name}  (split={split}, top_n={top_n}, smoke={smoke})')
    print('=' * 70)

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print(f'  Device: {device}')

    # ── Load queries ──────────────────────────────────────────────────────
    queries = load_queries(split, smoke=smoke)
    print(f'  Queries: {len(queries)}  (split={split})')

    # ── Load corpus ───────────────────────────────────────────────────────
    print(f'  Loading corpus: {corpus_path}')
    with open(corpus_path, encoding='utf-8') as f:
        corpus_dict = json.load(f)
    corpus = list(iter_corpus(corpus_dict))
    if smoke:
        corpus = corpus[:200]
    print(f'  Corpus: {len(corpus):,} pages')

    # ── Load model ────────────────────────────────────────────────────────
    model, fmt = load_model(checkpoint_path)
    print(f'  Model format: {fmt}')
    tokenizer = BertTokenizer.from_pretrained('BAAI/bge-base-en')
    model.eval().to(device)

    # ── Encode corpus (one-time, expensive) ───────────────────────────────
    print(f'  Encoding corpus ({len(corpus):,} pages)...')
    t0 = time.time()
    corpus_texts = [text for (_, _, text) in corpus]
    corpus_keys = [(doc, pn) for (doc, pn, _) in corpus]
    corpus_embs = encode_texts(model, tokenizer, corpus_texts, device,
                               batch_size=32)
    t_corpus = time.time() - t0
    print(f'    Done in {t_corpus:.1f}s ({len(corpus) / max(t_corpus, 1e-3):.1f} pages/s).')

    # ── Encode queries ────────────────────────────────────────────────────
    print(f'  Encoding {len(queries)} queries...')
    t0 = time.time()
    q_texts = [r['key_content']['question'] for r in queries]
    q_embs = encode_texts(model, tokenizer, q_texts, device, batch_size=32)
    t_query = time.time() - t0
    print(f'    Done in {t_query:.1f}s.')

    # ── Retrieve top-n per query ──────────────────────────────────────────
    print(f'  Retrieving top-{top_n} per query...')
    t0 = time.time()
    indices, scores = retrieve_top_n(q_embs, corpus_embs, n=top_n)
    t_retrieve = time.time() - t0
    print(f'    Done in {t_retrieve:.2f}s.')

    # ── Compute metrics per query, then aggregate ─────────────────────────
    per_query = []
    skipped = 0
    for qi, record in enumerate(queries):
        gold = golden_pages(record)
        if not gold:
            skipped += 1
            continue
        retrieved = [corpus_keys[int(idx)] for idx in indices[qi].tolist()]
        per_query.append({
            'qi': qi,
            'company': record.get('other_info', {}).get('company', '?'),
            'n_golden': len(gold),
            'hit@1':  hit_at_k(retrieved, gold, 1),
            'hit@10': hit_at_k(retrieved, gold, 10),
            'em':     em_at_n(retrieved, gold, n=top_n),
            'mrr':    mrr_for_query(retrieved, gold),
            'map':    average_precision(retrieved, gold),
            'ndcg':   ndcg_at_k(retrieved, gold, k=10),
        })

    if not per_query:
        raise RuntimeError('All queries skipped — no golden evidence found.')

    n = len(per_query)
    aggregate = {
        'hit@1':  sum(q['hit@1'] for q in per_query) / n * 100,
        'hit@10': sum(q['hit@10'] for q in per_query) / n * 100,
        'em':     sum(q['em']    for q in per_query) / n * 100,
        'mrr':    sum(q['mrr']   for q in per_query) / n * 100,
        'map':    sum(q['map']   for q in per_query) / n * 100,
        'ndcg':   sum(q['ndcg']  for q in per_query) / n * 100,
    }

    # ── Pretty print ──────────────────────────────────────────────────────
    print()
    print(f'=== Results: {name} (split={split}, n={n} queries, '
          f'{skipped} skipped) ===')
    print(f'  Hit@1   = {aggregate["hit@1"]:6.2f}')
    print(f'  Hit@10  = {aggregate["hit@10"]:6.2f}')
    print(f'  EM      = {aggregate["em"]:6.2f}')
    print(f'  MRR     = {aggregate["mrr"]:6.2f}')
    print(f'  MAP     = {aggregate["map"]:6.2f}')
    print(f'  NDCG    = {aggregate["ndcg"]:6.2f}')
    print()

    # ── Save JSON ─────────────────────────────────────────────────────────
    os.makedirs(output_dir, exist_ok=True)
    out_path = os.path.join(output_dir,
                            f'eval_output_{name}_{split}{"_smoke" if smoke else ""}.json')
    out_payload = {
        'name': name,
        'split': split,
        'smoke': smoke,
        'checkpoint': checkpoint_path,
        'model_format': fmt,
        'top_n': top_n,
        'corpus_size': len(corpus),
        'n_queries_evaluated': n,
        'n_queries_skipped': skipped,
        'timing_seconds': {
            'corpus_encode': round(t_corpus, 2),
            'query_encode':  round(t_query, 2),
            'retrieve':      round(t_retrieve, 2),
        },
        'aggregate': aggregate,
        'per_query': per_query,
    }
    with open(out_path, 'w', encoding='utf-8') as f:
        json.dump(out_payload, f, indent=2, ensure_ascii=False)
    print(f'  JSON saved: {out_path}')
    return out_payload


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--checkpoint', default=None,
                   help='LoRA-only or full BertModel state_dict .bin (omit for pretrained)')
    p.add_argument('--name', default=None,
                   help='display label (default: filename or pretrained)')
    p.add_argument('--split', choices=['val', 'test'], default='val')
    here = os.path.dirname(os.path.abspath(__file__))
    p.add_argument('--corpus',
                   default=os.path.join(here, 'paper_test_data', 'test_corpus.json'))
    p.add_argument('--top-n', type=int, default=100,
                   help='retrieve top-n per query (default: 100)')
    p.add_argument('--output-dir',
                   default=os.path.join(here, 'paper_test_data', 'eval_outputs'))
    p.add_argument('--smoke', action='store_true',
                   help='run on first 5 queries × first 200 corpus pages')
    args = p.parse_args()

    # Make `flgo.benchmark.fedrag_classification.config` importable for LoRA load
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

    name = args.name or (
        os.path.basename(args.checkpoint).replace('.bin', '')
        if args.checkpoint else 'pretrained'
    )
    run_eval(
        checkpoint_path=args.checkpoint,
        name=name,
        split=args.split,
        corpus_path=args.corpus,
        top_n=args.top_n,
        smoke=args.smoke,
        output_dir=args.output_dir,
    )


if __name__ == '__main__':
    main()
