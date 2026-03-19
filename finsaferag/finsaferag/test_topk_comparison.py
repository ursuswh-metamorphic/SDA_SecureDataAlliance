"""
Test medical RAG with different retrieval top_k values
Measures how F1/EM improve as we reduce the number of retrieved documents
"""

import os
import json
import sys
import warnings
import random
import numpy as np
import torch
import argparse
from pathlib import Path

os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'

from llama_index.core import Settings, Document
from llms.llm import get_llm
from index import get_index
from embs.embedding import get_embedding
from config import Config
from retriever import get_retriver
from llama_index.core.query_engine import RetrieverQueryEngine
from eval.evaluate_TRT import EvaluationResult_TRT, evaluating_TRT


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True


seed_everything(42)

# ========================================
# LOAD DATA
# ========================================

script_dir = os.path.dirname(os.path.abspath(__file__))
corpus_path = os.path.join(script_dir, "data/rag_corpus.json")
test_data_path = os.path.join(script_dir, "data/medical_test_50.json")

print(f"\n{'='*70}")
print("MEDICAL RAG - TOP_K COMPARISON TEST")
print(f"{'='*70}\n")

cfg = Config()

# Load corpus
print("[1/3] Loading corpus...")
with open(corpus_path, 'r', encoding='utf-8') as f:
    corpus_data = json.load(f)
print(f"      → Loaded {len(corpus_data)} documents")

# Load test data
print("[2/3] Loading test data...")
with open(test_data_path, 'r', encoding='utf-8') as f:
    test_queries = json.load(f)
print(f"      → Loaded {len(test_queries)} test queries")

# Load embedding & LLM
print("[3/3] Loading models...")
embeddings = get_embedding("BAAI/bge-base-en-v1.5")
Settings.embed_model = embeddings
Settings.llm = get_llm(cfg.llm)
print("      → Models loaded")

# Create documents
documents = [
    Document(
        id_=doc.get("id"),
        text=doc.get("text"),
        metadata={"id": doc.get("id"), "title": doc.get("title")}
    )
    for doc in corpus_data
]

# ========================================
# TEST DIFFERENT TOP_K VALUES
# ========================================

top_k_values = [1, 2, 3, 5, 10]
results_by_k = {}

for top_k in top_k_values:
    print(f"\n{'='*70}")
    print(f"Testing with top_k = {top_k}")
    print(f"{'='*70}\n")
    
    # Build index
    print(f"Building index...")
    index = get_index(documents, cfg)
    print(f"✓ Index built")
    
    # Create retriever with specific top_k
    retriever = get_retriver(
        cfg.retriever, 
        index, 
        cfg.retriever_mode,
        cfg=cfg
    )
    
    # Override similarity_top_k if possible
    if hasattr(retriever, 'similarity_top_k'):
        retriever.similarity_top_k = top_k
    if hasattr(retriever, '_similarity_top_k'):
        retriever._similarity_top_k = top_k
    
    # Create query engine
    query_engine = RetrieverQueryEngine.from_args(
        retriever=retriever,
        llm=Settings.llm
    )
    
    # Evaluate on test set
    evaluateResults_TRT = EvaluationResult_TRT()
    
    for idx, test_entry in enumerate(test_queries):
        key_content = test_entry.get('key_content', {})
        question = key_content.get('question', '')
        golden_ids = key_content.get('reference_idx', [])
        
        if not question:
            continue
        
        # Retrieve
        retrieved_nodes = query_engine.retrieve(question)
        retrieval_ids = [node.metadata.get('id', '') for node in retrieved_nodes]
        
        # Evaluate
        eval_result = evaluating_TRT(retrieval_ids, golden_ids)
        evaluateResults_TRT.add(eval_result)
        
        if (idx + 1) % 10 == 0:
            print(f"  Processed {idx + 1}/{len(test_queries)} queries")
    
    # Store results
    results_by_k[top_k] = {
        'F1': evaluateResults_TRT.results['F1'] / evaluateResults_TRT.results['n'],
        'EM': evaluateResults_TRT.results['em'] / evaluateResults_TRT.results['n'],
        'Hit@1': evaluateResults_TRT.results['hit1'] / evaluateResults_TRT.results['n'],
        'Hit@10': evaluateResults_TRT.results['hit10'] / evaluateResults_TRT.results['n'],
        'MRR': evaluateResults_TRT.results['mrr'] / evaluateResults_TRT.results['n'],
        'NDCG': evaluateResults_TRT.results['NDCG'] / evaluateResults_TRT.results['n'],
    }
    
    print(f"\nResults for top_k={top_k}:")
    print(f"  F1:     {results_by_k[top_k]['F1']:.1%}")
    print(f"  EM:     {results_by_k[top_k]['EM']:.1%}")
    print(f"  Hit@1:  {results_by_k[top_k]['Hit@1']:.1%}")
    print(f"  MRR:    {results_by_k[top_k]['MRR']:.1%}")
    print(f"  NDCG:   {results_by_k[top_k]['NDCG']:.1%}")


# ========================================
# PRINT SUMMARY TABLE
# ========================================

print(f"\n{'='*70}")
print("COMPARISON TABLE")
print(f"{'='*70}\n")

print(f"{'top_k':<8} {'F1':<12} {'EM':<12} {'Hit@1':<12} {'MRR':<12} {'NDCG':<12}")
print("-" * 70)

for top_k in sorted(results_by_k.keys()):
    r = results_by_k[top_k]
    print(
        f"{top_k:<8} "
        f"{r['F1']:>10.1%}  "
        f"{r['EM']:>10.1%}  "
        f"{r['Hit@1']:>10.1%}  "
        f"{r['MRR']:>10.1%}  "
        f"{r['NDCG']:>10.1%}"
    )

# Analysis
print(f"\n{'='*70}")
print("ANALYSIS")
print(f"{'='*70}\n")

best_f1_k = max(results_by_k.keys(), key=lambda k: results_by_k[k]['F1'])
best_mrr_k = max(results_by_k.keys(), key=lambda k: results_by_k[k]['MRR'])

print(f"Best F1: top_k={best_f1_k} with F1={results_by_k[best_f1_k]['F1']:.1%}")
print(f"Best MRR: top_k={best_mrr_k} with MRR={results_by_k[best_mrr_k]['MRR']:.1%}")

print(f"\nRecommendation:")
if best_f1_k == 1:
    print(f"  - Use top_k=1 for best precision (F1={results_by_k[1]['F1']:.1%})")
elif best_f1_k == 2:
    print(f"  - Use top_k=2 for balanced precision/recall (F1={results_by_k[2]['F1']:.1%})")
elif best_f1_k == 3:
    print(f"  - Use top_k=3 for good coverage (F1={results_by_k[3]['F1']:.1%})")
else:
    print(f"  - Use top_k={best_f1_k} (F1={results_by_k[best_f1_k]['F1']:.1%})")

print(f"\nNote: Lower top_k improves F1 because fewer wrong documents are retrieved")
print(f"      with single-document-per-query evaluation format.")
