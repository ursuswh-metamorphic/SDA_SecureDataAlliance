"""
regenerate_training_data.py — Phase 1 of `improve-fede-recipe-paper-rag-ft-kd-gle.md`.

Replaces the under-diversified `selected_data.json` (5 companies in our local
copy) with self-supervised (query, reference) pairs sampled from the FULL
368-doc `train_corpus.json` distributed by the FedE4RAG paper.

Strategy: split each page into ~150-word chunks, then for each page emit
N=2 contrastive pairs where:
  * `question` = a chunk from the page (acts as the query at retrieval time)
  * `reference` = ANOTHER chunk from the SAME page (the target to retrieve)

Why this works for paper-faithful retrieval training:
  - Each pair has a unique reference text → in-batch InfoNCE has clean negatives
  - Both chunks share topic/terminology of the page → contrastive signal is meaningful
  - Page-level disjoint with eval corpus (verified: 0 page overlap with test_corpus.json)
    → no train-test leak
  - 368 docs × ~63 pages × 2 pairs ≈ 45K pairs covering ALL companies in eval

Schema matches what core.py:60-77 expects:
  [{company, page, index, reference, question}, ...]

Usage (run from repo root):
  python FedE/scripts/regenerate_training_data.py [--out PATH] [--pairs-per-page N] [--seed S]

Default: writes FedE/selected_data.json with 2 pairs per page, seed=42.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import re
from collections import Counter
from typing import Dict, List


HERE = os.path.dirname(os.path.abspath(__file__))
FEDE_DIR = os.path.dirname(HERE)
DEFAULT_TRAIN_CORPUS = os.path.join(FEDE_DIR, "train_corpus.json")
DEFAULT_OUT = os.path.join(FEDE_DIR, "selected_data.json")


# ── Helpers ────────────────────────────────────────────────────────────────

_SENTENCE_SPLITTER = re.compile(r'(?<=[.!?])\s+')


def chunk_text(text: str, words_per_chunk: int = 150,
               min_words: int = 30) -> List[str]:
    """Split text into ~words_per_chunk-word chunks at sentence boundaries.

    Skips final chunk if it has fewer than `min_words` words (avoids tiny
    fragments that would be poor retrieval targets).
    """
    text = text.replace('\n', ' ').strip()
    sentences = [s.strip() for s in _SENTENCE_SPLITTER.split(text) if s.strip()]
    if not sentences:
        return []

    chunks: List[str] = []
    cur: List[str] = []
    cur_word_count = 0
    for s in sentences:
        s_words = len(s.split())
        if cur_word_count + s_words > words_per_chunk and cur:
            chunks.append(' '.join(cur))
            cur = [s]
            cur_word_count = s_words
        else:
            cur.append(s)
            cur_word_count += s_words

    if cur:
        last_chunk = ' '.join(cur)
        if len(last_chunk.split()) >= min_words or not chunks:
            chunks.append(last_chunk)

    return chunks


def stable_index(doc_name: str, page_num: str, k: int) -> int:
    """Deterministic int index from (doc, page, pair_idx) for traceability."""
    h = hashlib.md5(f'{doc_name}_{page_num}_{k}'.encode()).hexdigest()
    return int(h[:8], 16) % (10 ** 8)


def doc_to_company(doc_name: str) -> str:
    """Extract company token from doc_name like 'PEPSICO_2022_10K' → 'PEPSICO'."""
    return doc_name.split('_')[0]


def generate_pairs_from_page(doc_name: str, page_num: str, page_text: str,
                              n_pairs: int, rng: random.Random) -> List[Dict]:
    """Generate up to `n_pairs` (query, reference) pairs from one page.

    Each pair samples two DISTINCT chunks from the page:
      question = chunk_i, reference = chunk_j  (i != j).
    Returns [] if page is too short to chunk into 2+ pieces.
    """
    chunks = chunk_text(page_text, words_per_chunk=150, min_words=30)
    if len(chunks) < 2:
        return []

    n_to_emit = min(n_pairs, len(chunks) * (len(chunks) - 1))
    seen = set()
    pairs: List[Dict] = []
    company = doc_to_company(doc_name)

    # Cap retries to avoid pathological infinite loops on tiny chunk sets
    max_attempts = 5 * n_to_emit + 10
    attempt = 0
    while len(pairs) < n_to_emit and attempt < max_attempts:
        attempt += 1
        i = rng.randrange(len(chunks))
        j = rng.randrange(len(chunks))
        if i == j or (i, j) in seen:
            continue
        seen.add((i, j))
        pairs.append({
            'company':   company,
            'page':      doc_name + '#p' + str(page_num),  # compact, identifies origin
            'index':     stable_index(doc_name, str(page_num), len(pairs)),
            'reference': chunks[j],
            'question':  chunks[i],
        })

    return pairs


# ── Main ───────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--corpus', default=DEFAULT_TRAIN_CORPUS,
                    help='train_corpus.json path (368 docs)')
    ap.add_argument('--out', default=DEFAULT_OUT,
                    help='output selected_data.json path')
    ap.add_argument('--pairs-per-page', type=int, default=2,
                    help='max contrastive pairs to generate per page (default 2)')
    ap.add_argument('--seed', type=int, default=42,
                    help='RNG seed for reproducibility')
    ap.add_argument('--max-pairs', type=int, default=None,
                    help='cap total pairs (default: no cap)')
    args = ap.parse_args()

    rng = random.Random(args.seed)

    print(f'[regenerate] Loading corpus: {args.corpus}')
    with open(args.corpus, encoding='utf-8') as f:
        corpus: Dict[str, Dict] = json.load(f)
    print(f'[regenerate] {len(corpus)} docs in corpus')

    all_pairs: List[Dict] = []
    n_pages_seen = 0
    n_pages_skipped = 0

    for doc_name, pages in corpus.items():
        if not isinstance(pages, dict):
            continue
        for page_num, page_data in pages.items():
            n_pages_seen += 1
            text = (page_data.get('page_content', '')
                    if isinstance(page_data, dict)
                    else str(page_data))
            if not text or len(text.strip()) < 200:
                n_pages_skipped += 1
                continue
            pairs = generate_pairs_from_page(
                doc_name, page_num, text, args.pairs_per_page, rng,
            )
            if not pairs:
                n_pages_skipped += 1
                continue
            all_pairs.extend(pairs)

    rng.shuffle(all_pairs)

    if args.max_pairs and len(all_pairs) > args.max_pairs:
        all_pairs = all_pairs[:args.max_pairs]

    # ── Write output ─────────────────────────────────────────────────
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, 'w', encoding='utf-8') as f:
        json.dump(all_pairs, f, ensure_ascii=False)

    # ── Report stats ─────────────────────────────────────────────────
    out_size_mb = os.path.getsize(args.out) / 1024 / 1024
    company_counter = Counter(p['company'] for p in all_pairs)
    doc_counter = Counter(p['page'].split('#p')[0] for p in all_pairs)

    print()
    print('=' * 70)
    print(f'[regenerate] DONE')
    print('=' * 70)
    print(f'  Pages seen:       {n_pages_seen:,}')
    print(f'  Pages skipped:    {n_pages_skipped:,} '
          f'(too short / single chunk)')
    print(f'  Pairs generated:  {len(all_pairs):,}')
    print(f'  Unique companies: {len(company_counter)} '
          f'(was 5 in old data_50000_random.json)')
    print(f'  Unique docs:      {len(doc_counter)}')
    print(f'  Output size:      {out_size_mb:.1f} MB → {args.out}')
    print()
    print('  Top-10 company distribution:')
    for co, n in company_counter.most_common(10):
        print(f'    {co:<25} {n:>6} pairs')
    if len(company_counter) > 10:
        rest = sum(n for _, n in company_counter.most_common()[10:])
        print(f'    {"... " + str(len(company_counter) - 10) + " more cos":<25} {rest:>6} pairs')


if __name__ == '__main__':
    main()
