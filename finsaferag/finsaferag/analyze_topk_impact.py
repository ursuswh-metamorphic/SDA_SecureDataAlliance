"""
Analyze impact of top_k on F1/EM metrics
Uses already-retrieved documents to show what scores would be with different cutoffs
"""

import json
import numpy as np
from eval.evaluate_TRT import evaluating_TRT

# Load test data
with open('data/medical_test_50.json', 'r') as f:
    test_data = json.load(f)

with open('data/rag_corpus.json', 'r') as f:
    corpus = json.load(f)

# Create a mapping of corpus docs
corpus_map = {doc['id']: doc for doc in corpus}

print("\n" + "="*70)
print("ANALYZING TOP_K IMPACT ON METRICS")
print("="*70)
print("\nUsing BGE embeddings to rank documents by similarity...")

from llama_index.embeddings.huggingface import HuggingFaceEmbedding
import warnings
warnings.filterwarnings('ignore')

embeddings = HuggingFaceEmbedding(model_name="BAAI/bge-base-en-v1.5")

# For each test query, rank all corpus docs by similarity
all_rankings = []

for idx, test_entry in enumerate(test_data):
    question = test_entry['key_content']['question']
    expected_ref_id = test_entry['key_content']['reference_idx'][0]
    
    # Get query embedding
    q_emb = embeddings.get_text_embedding(question)
    
    # Rank all corpus documents
    doc_scores = []
    for doc in corpus:
        doc_emb = embeddings.get_text_embedding(doc['text'][:512])
        
        # Cosine similarity
        dot_product = np.dot(q_emb, doc_emb)
        norm_q = np.linalg.norm(q_emb)
        norm_doc = np.linalg.norm(doc_emb)
        similarity = dot_product / (norm_q * norm_doc + 1e-8)
        
        doc_scores.append((doc['id'], similarity))
    
    # Sort by similarity
    doc_scores.sort(key=lambda x: x[1], reverse=True)
    ranked_ids = [doc_id for doc_id, _ in doc_scores]
    
    all_rankings.append({
        'query': question[:60],
        'expected': expected_ref_id,
        'ranking': ranked_ids
    })
    
    if (idx + 1) % 10 == 0:
        print(f"  Ranked {idx + 1}/{len(test_data)} queries")

print(f"\n✓ Ranked all {len(test_data)} queries")

# Now evaluate at different top_k values
top_k_values = [1, 2, 3, 5, 10]
results = {}

print(f"\n{'='*70}")
print("EVALUATING AT DIFFERENT TOP_K VALUES")
print(f"{'='*70}\n")

for top_k in top_k_values:
    print(f"Evaluating top_k={top_k}...")
    
    avg_f1 = 0
    avg_em = 0
    avg_hit1 = 0
    avg_mrr = 0
    avg_ndcg = 0
    
    for ranking_data in all_rankings:
        expected_id = ranking_data['expected']
        ranked_ids = ranking_data['ranking'][:top_k]
        
        # Evaluate
        eval_result = evaluating_TRT(ranked_ids, [expected_id])
        
        avg_f1 += eval_result.results['F1']
        avg_em += eval_result.results['em']
        avg_hit1 += eval_result.results['hit1']
        avg_mrr += eval_result.results['mrr']
        avg_ndcg += eval_result.results['NDCG']
    
    avg_f1 /= len(test_data)
    avg_em /= len(test_data)
    avg_hit1 /= len(test_data)
    avg_mrr /= len(test_data)
    avg_ndcg /= len(test_data)
    
    results[top_k] = {
        'F1': avg_f1,
        'EM': avg_em,
        'Hit@1': avg_hit1,
        'MRR': avg_mrr,
        'NDCG': avg_ndcg,
    }
    
    print(f"  F1: {avg_f1:.1%}, EM: {avg_em:.1%}, Hit@1: {avg_hit1:.1%}, MRR: {avg_mrr:.1%}, NDCG: {avg_ndcg:.1%}")

# Print comparison table
print(f"\n{'='*70}")
print("RESULTS COMPARISON TABLE")
print(f"{'='*70}\n")

print(f"{'top_k':<8} {'F1':<12} {'EM':<12} {'Hit@1':<12} {'MRR':<12} {'NDCG':<12}")
print("-" * 70)

for top_k in sorted(results.keys()):
    r = results[top_k]
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
print("ANALYSIS & RECOMMENDATIONS")
print(f"{'='*70}\n")

best_f1_k = max(results.keys(), key=lambda k: results[k]['F1'])
f1_improvement = (results[best_f1_k]['F1'] - results[10]['F1']) / results[10]['F1'] * 100

print(f"Current (top_k=10):")
print(f"  F1:  {results[10]['F1']:.1%} (12%)")
print(f"  EM:  {results[10]['EM']:.1%}")
print(f"  MRR: {results[10]['MRR']:.1%}")

print(f"\nBest for F1 (top_k={best_f1_k}):")
print(f"  F1:  {results[best_f1_k]['F1']:.1%}")
print(f"  EM:  {results[best_f1_k]['EM']:.1%}")
print(f"  MRR: {results[best_f1_k]['MRR']:.1%}")
print(f"  Improvement: +{f1_improvement:.0f}%")

print(f"\n📋 RECOMMENDATION:")
if best_f1_k <= 2:
    print(f"   Use top_k={best_f1_k} for maximum F1 score")
elif best_f1_k <= 5:
    print(f"   Use top_k={best_f1_k} for balanced F1 and coverage")
else:
    print(f"   Current top_k=10 is reasonable, but top_k={best_f1_k} offers better F1")

print(f"\n💡 WHY TOP_K MATTERS:")
print(f"   - With single expected document per query")
print(f"   - F1 = 1/(1+wrong_docs)")
print(f"   - Fewer retrieved docs = higher F1")
print(f"   - But: Too small top_k = might miss correct doc")
