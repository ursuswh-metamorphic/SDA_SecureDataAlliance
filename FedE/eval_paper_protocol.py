"""
eval_paper_protocol.py — Replica chính xác eval protocol của paper FedE4RAG.

Phát hiện audit (2026-05-17, commit DocAILab/FedE4RAG main):
1. `RAGTest/data/loader.py`: corpus = first 6066 pages của test_corpus.json
   + **APPEND val_qa references** (text + reference_idx) AS extra Documents.
2. `RAGTest/index.py`: SentenceSplitter(chunk_size=cfg.chunk_size, chunk_overlap=20).
   cfg.chunk_size = 2048 (config.toml).
3. `RAGTest/main_50_test.py`: query_expansion + similarity_top_k=10
   → retrieve top-10 chunks per query.
4. `RAGTest/eval/evaluate_rag.py:514-519`: Paper's Hit definition là
   "ANY retrieved id ∈ first-K golden" — equivalent standard Recall@K.

Script này replica chính xác setup paper để measure pipeline trên cùng metric.
Important: chunk_id matching uses `metadata['id']` từ Document (paper's chunk indexing).

Usage:
    python FedE/eval_paper_protocol.py --checkpoint X.bin --split val
    python FedE/eval_paper_protocol.py --split val          # pretrained baseline
    python FedE/eval_paper_protocol.py --checkpoint X.bin --split val --append-refs

CPU-runnable nhưng GPU faster. Khoảng 5-10 min/eval trên CPU (BGE-base).
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
from typing import Iterable, List, Tuple

import numpy as np
import torch
import torch.nn.functional as F


# ══════════════════════════════════════════════════════════════════════════════
#   Paper's Hit/Mrr — replicate from RAGTest/eval/evaluate_rag.py:514-519
# ══════════════════════════════════════════════════════════════════════════════

def paper_hit(retrieved_ids: List, expected_ids: List) -> float:
    """Paper's Hit function (evaluate_rag.py:514-519):

      def Hit(retrieved_ids, expected_ids):
          is_hit = any(id in expected_ids for id in retrieved_ids)
          return 1.0 if is_hit else 0.0

    Returns 1.0 if ANY retrieved id is in expected_ids (the truncated golden set).
    """
    if not retrieved_ids or not expected_ids:
        return 0.0
    expected_set = set(expected_ids)
    is_hit = any(rid in expected_set for rid in retrieved_ids)
    return 1.0 if is_hit else 0.0


def paper_mrr(retrieved_ids: List, expected_ids: List) -> float:
    """Paper's Mrr function (evaluate_rag.py:506-513):

      def Mrr(retrieved_ids, expected_ids):
          for i, id in enumerate(retrieved_ids):
              if id in expected_ids:
                  return 1.0 / (i + 1)
          return 0.0
    """
    if not retrieved_ids or not expected_ids:
        return 0.0
    expected_set = set(expected_ids)
    for i, rid in enumerate(retrieved_ids):
        if rid in expected_set:
            return 1.0 / (i + 1)
    return 0.0


# ══════════════════════════════════════════════════════════════════════════════
#   Sentence-aware chunking (replicate LlamaIndex SentenceSplitter behavior)
# ══════════════════════════════════════════════════════════════════════════════

_SENTENCE_RE = re.compile(r'(?<=[.!?])\s+')


def sentence_split_to_chunks(text: str, chunk_size: int = 2048,
                              chunk_overlap: int = 20) -> List[str]:
    """Split text into chunks of ~chunk_size characters at sentence boundaries.

    Approximates LlamaIndex SentenceSplitter(chunk_size, chunk_overlap).
    chunk_size here is in CHARACTERS (LlamaIndex uses tokens; conversion ≈ 4
    chars/token English → chunk_size=2048 tokens ≈ 8192 chars). We use 2048
    chars as conservative — most BGE input gets truncated at 512 tokens
    (~2048 chars) anyway.
    """
    text = text.strip()
    if not text:
        return []
    if len(text) <= chunk_size:
        return [text]

    sentences = _SENTENCE_RE.split(text)
    chunks: List[str] = []
    cur: List[str] = []
    cur_len = 0

    for sent in sentences:
        sent_len = len(sent) + 1  # account for space
        if cur_len + sent_len > chunk_size and cur:
            chunks.append(' '.join(cur))
            # Overlap: keep last sentence(s) up to chunk_overlap chars
            overlap_text = ' '.join(cur[-1:])[:chunk_overlap]
            cur = [overlap_text, sent] if overlap_text else [sent]
            cur_len = len(overlap_text) + sent_len
        else:
            cur.append(sent)
            cur_len += sent_len

    if cur:
        chunks.append(' '.join(cur))
    return chunks


# ══════════════════════════════════════════════════════════════════════════════
#   Build corpus theo paper's loader.py
# ══════════════════════════════════════════════════════════════════════════════

def build_paper_corpus(test_corpus_path: str, qa_path: str,
                       corpus_cap: int = 6066,
                       append_refs: bool = True) -> List[Tuple[int, str, str]]:
    """Replicate RAGTest/data/loader.py:get_documents() behavior.

    Returns: list of (chunk_id, text, source) tuples ready to index.

    Steps:
      1. Load test_corpus.json (dict of doc_name → {page_num: page_dict}).
      2. Iterate pages, take first `corpus_cap` (paper uses 6066). Each page
         becomes Document with metadata['id'] = passage['index'].
      3. (If append_refs) Load qa_path (data_50 / data_100), iterate entries,
         for each (reference, reference_idx) pair append a Document with
         text=reference, metadata['id']=reference_idx.
      4. Sentence-split each Document into chunks (LlamaIndex default chunk_size).
         Each chunk inherits parent metadata['id'].
      5. Return flat list of (id, chunk_text, source) tuples.
    """
    documents = []  # list of (id, text, source)

    # Step 1-2: Load corpus
    print(f'[build_corpus] Load {test_corpus_path}')
    with open(test_corpus_path, encoding='utf-8') as f:
        corpus = json.load(f)

    n_pages = 0
    for doc_name, pages in corpus.items():
        for page_num, page_data in pages.items():
            if not isinstance(page_data, dict):
                continue
            text = page_data.get('page_content', '')
            chunk_id = page_data.get('index')
            if chunk_id is None:
                continue
            if not text or not text.strip():
                continue
            documents.append((chunk_id, text, f'{doc_name}#p{page_num}'))
            n_pages += 1
            if n_pages == corpus_cap:
                break
        if n_pages == corpus_cap:
            break
    print(f'[build_corpus] Loaded {n_pages} corpus pages (capped at {corpus_cap})')

    # Step 3: Append references from qa data (paper does this!)
    if append_refs:
        print(f'[build_corpus] Load {qa_path} and append references')
        with open(qa_path, encoding='utf-8') as f:
            qa = json.load(f)
        n_refs = 0
        for entry in qa:
            kc = entry.get('key_content', {})
            refs = kc.get('reference', [])
            ref_ids = kc.get('reference_idx', [])
            for ref_text, ref_id in zip(refs, ref_ids):
                if not ref_text or not ref_text.strip():
                    continue
                documents.append((ref_id, ref_text,
                                  f'{entry.get("other_info", {}).get("doc_name", "?")}#ref'))
                n_refs += 1
        print(f'[build_corpus] Appended {n_refs} reference docs')
    else:
        print(f'[build_corpus] SKIP append references (--no-append-refs)')

    # Step 4: Chunk each doc
    print(f'[build_corpus] Chunking with sentence-split (chunk_size=2048 chars)')
    chunks = []
    for chunk_id, text, source in documents:
        for chunk_text in sentence_split_to_chunks(text, chunk_size=2048,
                                                    chunk_overlap=20):
            chunks.append((chunk_id, chunk_text, source))
    print(f'[build_corpus] Total chunks: {len(chunks)}')

    return chunks


# ══════════════════════════════════════════════════════════════════════════════
#   Embedding helpers
# ══════════════════════════════════════════════════════════════════════════════

def encode_texts(model, tokenizer, texts: List[str], device,
                 batch_size: int = 32, max_length: int = 512) -> torch.Tensor:
    """Mean-pool encode + L2 normalize."""
    embs = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i + batch_size]
        inp = tokenizer(batch, return_tensors='pt', padding=True,
                        truncation=True, max_length=max_length).to(device)
        with torch.no_grad():
            out = model(**inp)
        emb = out.last_hidden_state.mean(dim=1)
        embs.append(emb.cpu())
        if (i // batch_size) % 20 == 0 and i > 0:
            print(f'    encoded {i + len(batch):,}/{len(texts):,}')
    e = torch.cat(embs, dim=0)
    return F.normalize(e, dim=-1, p=2)


def load_model(checkpoint_path: str | None):
    """Load BGE-base, apply LoRA checkpoint if given."""
    from transformers import BertModel
    base = BertModel.from_pretrained('BAAI/bge-base-en')
    if not checkpoint_path or not os.path.exists(checkpoint_path):
        if checkpoint_path:
            print(f'  [load_model] checkpoint not found: {checkpoint_path}')
        return base, 'pretrained'

    print(f'  [load_model] Loading {checkpoint_path}')
    state = torch.load(checkpoint_path, map_location='cpu', weights_only=True)
    has_lora = any('lora' in k.lower() for k in state.keys())
    if has_lora:
        from peft import LoraConfig, get_peft_model
        # Import via path to avoid flgo init
        sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
        from flgo.benchmark.fedrag_classification.config import (
            LORA_R, LORA_ALPHA, LORA_TARGETS, LORA_DROPOUT,
        )
        cfg = LoraConfig(r=LORA_R, lora_alpha=LORA_ALPHA,
                         target_modules=LORA_TARGETS, lora_dropout=LORA_DROPOUT,
                         bias='none')
        wrapped = get_peft_model(base, cfg)
        miss, unexp = wrapped.load_state_dict(state, strict=False)
        print(f'    LoRA-only ({len(state)} tensors); missing={len(miss)}, unexp={len(unexp)}')
        merged = wrapped.merge_and_unload()
        return merged, 'lora_merged'

    # Full state dict
    clean = {}
    for k, v in state.items():
        nk = k.replace('module.', '').replace('model.', '', 1) if 'model.' in k else k
        clean[nk] = v
    base.load_state_dict(clean, strict=False)
    return base, 'full_base'


# ══════════════════════════════════════════════════════════════════════════════
#   Main eval loop
# ══════════════════════════════════════════════════════════════════════════════

def run_paper_eval(checkpoint_path: str | None, name: str, split: str,
                   corpus_path: str, qa_path: str,
                   top_k: int = 10, corpus_cap: int = 6066,
                   append_refs: bool = True,
                   batch_size: int = 32,
                   output_dir: str | None = None) -> dict:
    """Run paper-protocol eval. Returns metrics dict + saves JSON."""
    from transformers import BertTokenizer

    print('=' * 70)
    print(f'  Paper-protocol eval: {name} (split={split}, top_k={top_k},')
    print(f'  corpus_cap={corpus_cap}, append_refs={append_refs})')
    print('=' * 70)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'  Device: {device}')

    # ── Build corpus per paper protocol ───────────────────────────────────
    chunks = build_paper_corpus(corpus_path, qa_path, corpus_cap=corpus_cap,
                                 append_refs=append_refs)
    chunk_ids = [c[0] for c in chunks]
    chunk_texts = [c[1] for c in chunks]

    # ── Load model + tokenizer ────────────────────────────────────────────
    model, fmt = load_model(checkpoint_path)
    print(f'  Model format: {fmt}')
    tokenizer = BertTokenizer.from_pretrained('BAAI/bge-base-en')
    model.eval().to(device)

    # ── Load queries ──────────────────────────────────────────────────────
    print(f'  Load queries: {qa_path}')
    with open(qa_path, encoding='utf-8') as f:
        queries = json.load(f)
    print(f'  Queries: {len(queries)}')

    # ── Encode corpus ─────────────────────────────────────────────────────
    print(f'  Encoding {len(chunks):,} chunks...')
    t0 = time.time()
    chunk_embs = encode_texts(model, tokenizer, chunk_texts, device, batch_size)
    t_corpus = time.time() - t0
    print(f'    Done in {t_corpus:.1f}s ({len(chunks) / max(t_corpus, 1e-3):.1f} chunks/s)')

    # ── Encode queries ────────────────────────────────────────────────────
    print(f'  Encoding {len(queries)} queries...')
    t0 = time.time()
    q_texts = [q['key_content']['question'] for q in queries]
    q_embs = encode_texts(model, tokenizer, q_texts, device, batch_size)
    t_q = time.time() - t0
    print(f'    Done in {t_q:.1f}s')

    # ── Retrieve top-k per query + compute paper Hit/MRR ──────────────────
    sims = q_embs @ chunk_embs.t()       # (Q, C)
    scores, indices = torch.topk(sims, k=min(top_k, len(chunks)), dim=-1)

    per_query = []
    for qi, q in enumerate(queries):
        retrieved_ids = [chunk_ids[int(i)] for i in indices[qi].tolist()]
        golden_ids = q['key_content']['reference_idx']

        hit1 = paper_hit(retrieved_ids, golden_ids[0:1])
        hit10 = paper_hit(retrieved_ids, golden_ids[0:10])
        mrr = paper_mrr(retrieved_ids, golden_ids)

        per_query.append({
            'qi': qi,
            'company': q.get('other_info', {}).get('company', '?'),
            'doc_name': q.get('other_info', {}).get('doc_name', '?'),
            'n_golden': len(golden_ids),
            'hit1': hit1,
            'hit10': hit10,
            'mrr': mrr,
            'retrieved_top5': retrieved_ids[:5],
            'golden_first': golden_ids[:3] if isinstance(golden_ids, list) else [],
        })

    n = len(per_query)
    agg = {
        'hit1': sum(q['hit1'] for q in per_query) / n * 100,
        'hit10': sum(q['hit10'] for q in per_query) / n * 100,
        'mrr': sum(q['mrr'] for q in per_query) / n * 100,
    }

    # ── Print ─────────────────────────────────────────────────────────────
    print()
    print(f'=== Paper-protocol results: {name} (split={split}, n={n}) ===')
    print(f'  Hit@1   = {agg["hit1"]:6.2f}   (paper def: any retrieved == first golden)')
    print(f'  Hit@10  = {agg["hit10"]:6.2f}   (paper def: any retrieved in first 10 golden)')
    print(f'  MRR     = {agg["mrr"]:6.2f}')
    print()

    # ── Save ──────────────────────────────────────────────────────────────
    payload = {
        'name': name,
        'split': split,
        'checkpoint': checkpoint_path,
        'model_format': fmt,
        'corpus_path': corpus_path,
        'qa_path': qa_path,
        'top_k': top_k,
        'corpus_cap': corpus_cap,
        'append_refs': append_refs,
        'n_chunks_indexed': len(chunks),
        'n_queries': n,
        'protocol': 'paper-faithful (loader.py append refs + Hit definition)',
        'aggregate': agg,
        'per_query': per_query,
    }
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        out_path = os.path.join(output_dir,
            f'paper_protocol_{name}_{split}{"_no_refs" if not append_refs else ""}.json')
        with open(out_path, 'w', encoding='utf-8') as f:
            json.dump(payload, f, indent=2, ensure_ascii=False)
        print(f'  JSON saved: {out_path}')

    return payload


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--checkpoint', default=None,
                   help='LoRA-only or full state_dict (default: pretrained BGE-base)')
    p.add_argument('--name', default=None,
                   help='Display label (default: filename or pretrained)')
    p.add_argument('--split', choices=['val', 'test'], default='val')
    here = os.path.dirname(os.path.abspath(__file__))
    p.add_argument('--corpus',
                   default=os.path.join(here, 'paper_test_data', 'test_corpus.json'))
    p.add_argument('--qa-val',
                   default=os.path.join(here, 'paper_test_data', 'val_qa_data_50.json'))
    p.add_argument('--qa-test',
                   default=os.path.join(here, 'paper_test_data', 'test_qa_data_100.json'))
    p.add_argument('--top-k', type=int, default=10,
                   help='Paper uses similarity_top_k=10 with query_expansion (default: 10)')
    p.add_argument('--corpus-cap', type=int, default=6066,
                   help='Paper caps at 6066 docs (data/loader.py)')
    p.add_argument('--no-append-refs', action='store_true',
                   help='SKIP appending qa reference texts to corpus '
                        '(ablation — paper DOES append; this tests if Hit lift is purely '
                        'from append-refs trick)')
    p.add_argument('--batch-size', type=int, default=32)
    p.add_argument('--output-dir',
                   default=os.path.join(here, 'paper_test_data', 'eval_outputs'))
    args = p.parse_args()

    name = args.name or (
        os.path.basename(args.checkpoint).replace('.bin', '')
        if args.checkpoint else 'pretrained'
    )
    qa_path = args.qa_val if args.split == 'val' else args.qa_test
    run_paper_eval(
        checkpoint_path=args.checkpoint,
        name=name,
        split=args.split,
        corpus_path=args.corpus,
        qa_path=qa_path,
        top_k=args.top_k,
        corpus_cap=args.corpus_cap,
        append_refs=not args.no_append_refs,
        batch_size=args.batch_size,
        output_dir=args.output_dir,
    )


if __name__ == '__main__':
    main()
