"""
Medical RAG Testing Script
========================
Tests medical RAG system using medical_test_50.json and rag_corpus.json
Uses same evaluation standards as main_50_test.py (TRT evaluation framework)
"""

import os
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'

import json
import sys
import warnings
from pathlib import Path
from llama_index.core import Settings, PromptTemplate, Document
from llms.llm import get_llm
from index import get_index
from embs.embedding import get_embedding
from config import Config
from retriever import response_synthesizer, query_expansion, get_retriver
from llama_index.core.query_engine import RetrieverQueryEngine
from process.postprocess_rerank import get_postprocessor
from eval.EvalModelAgent import EvalModelAgent
from eval.evaluate_TRT import EvaluationResult_TRT, evaluating_TRT
import random
import numpy as np
import torch
import argparse


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True

seed_everything(42)


# ========================================
# ARGUMENT PARSING
# ========================================

parser = argparse.ArgumentParser(description='Test Medical RAG System')
parser.add_argument('--model', type=str, default=None, help='Path to trained embedding model')
parser.add_argument('--corpus', type=str, default=None, help='Path to medical corpus (rag_corpus.json)')
parser.add_argument('--test_data', type=str, default=None, help='Path to medical test data (medical_test_50.json)')
parser.add_argument('--output_dir', type=str, default='./medical_results', help='Output directory for results')
args = parser.parse_args()

# Set defaults
script_dir = os.path.dirname(os.path.abspath(__file__))
if args.corpus is None:
    args.corpus = os.path.join(script_dir, "data/rag_corpus.json")

if args.test_data is None:
    args.test_data = os.path.join(script_dir, "data/medical_test_50.json")

print(f"\n{'='*60}")
print("MEDICAL RAG TESTING")
print(f"{'='*60}")
print(f"Corpus:     {args.corpus}")
print(f"Test Data:  {args.test_data}")
print(f"Model:      {args.model or 'Default BGE-en'}")
print(f"{'='*60}\n")


# ========================================
# SETUP CONFIG
# ========================================

cfg = Config()
os.makedirs(args.output_dir, exist_ok=True)


# ========================================
# LOAD EMBEDDING MODEL
# ========================================

print("[1/5] Loading embedding model...")
if args.model is None:
    print("      → Using default: BAAI/bge-base-en-v1.5")
    embeddings = get_embedding("BAAI/bge-base-en-v1.5")
    model_name = "BAAI_bge_base_en_v1.5"
else:
    print(f"      → Loading trained model")
    embeddings = get_embedding(args.model)
    model_name = os.path.basename(args.model)


# ========================================
# LOAD MEDICAL CORPUS
# ========================================

print("[2/5] Loading medical corpus...")
try:
    with open(args.corpus, 'r', encoding='utf-8') as f:
        corpus_data = json.load(f)
    print(f"      → Loaded {len(corpus_data)} documents")
except Exception as e:
    print(f"      ❌ Error loading corpus: {e}")
    sys.exit(1)

# Convert to LlamaIndex Documents
corpus_docs = []
for item in corpus_data:
    doc = Document(
        text=item.get('text', ''),
        metadata={
            'title': item.get('title', ''),
            'id': item.get('id', ''),
            'source': 'medical'
        },
        doc_id=item.get('id', '')
    )
    corpus_docs.append(doc)


# ========================================
# LOAD TEST DATA
# ========================================

print("[3/5] Loading medical test data...")
try:
    with open(args.test_data, 'r', encoding='utf-8') as f:
        test_data = json.load(f)
    
    # Handle both list and dict formats
    if isinstance(test_data, dict):
        test_queries = test_data.get('test_queries', []) or list(test_data.values())
    else:
        test_queries = test_data
    
    print(f"      → Loaded {len(test_queries)} test queries")
except Exception as e:
    print(f"      ❌ Error loading test data: {e}")
    sys.exit(1)


# ========================================
# BUILD INDEX
# ========================================

print("[4/5] Building FAISS index...")
Settings.chunk_size = cfg.chunk_size
Settings.llm = get_llm(cfg.llm)
Settings.embed_model = embeddings

persist_dir = f"{cfg.persist_dir}_medical_{model_name}"

try:
    index, hierarchical_storage_context = get_index(
        documents=corpus_docs,
        persist_dir=persist_dir,
        split_type=cfg.split_type,
        chunk_size=cfg.chunk_size
    )
    print(f"      → Index ready at: {persist_dir}")
except Exception as e:
    print(f"      ❌ Error building index: {e}")
    sys.exit(1)


# ========================================
# SETUP QUERY ENGINE
# ========================================

print("[5/5] Setting up query engine...")

query_engine = RetrieverQueryEngine(
    retriever=get_retriver(cfg.retriever, index, hierarchical_storage_context=hierarchical_storage_context),
    response_synthesizer=response_synthesizer(0),
    node_postprocessors=[get_postprocessor(cfg)]
)

text_qa_template_str = (
    "Below is medical context information.\n"
    "---------------------\n"
    "{context_str}\n"
    "---------------------\n"
    "Based solely on the above context, please answer the following medical question: {query_str}\n"
    "If the context does not contain relevant information, say 'Information not available in context'."
)
text_qa_template = PromptTemplate(text_qa_template_str)
query_engine.update_prompts({"response_synthesizer:text_qa_template": text_qa_template})

query_engine = query_expansion([query_engine], query_number=4, similarity_top_k=10)
query_engine = RetrieverQueryEngine.from_args(query_engine)

print("      → Query engine ready\n")


# ========================================
# RUN EVALUATION
# ========================================

print(f"{'='*60}")
print("STARTING MEDICAL RAG EVALUATION (TRT Framework)")
print(f"{'='*60}\n")

evaluateResults_TRT = EvaluationResult_TRT()
evalAgent = EvalModelAgent(cfg)
all_num = 0

for idx, test_entry in enumerate(test_queries):
    idx_display = idx + 1
    
    try:
        # Extract query info
        key_content = test_entry.get('key_content', {})
        other_info = test_entry.get('other_info', {})
        
        question = key_content.get('question', '')
        expected_answer = key_content.get('answer', '')
        golden_context = key_content.get('reference', [])
        golden_ids = key_content.get('reference_idx', [])  # Use reference_idx as golden_ids
        
        if not question:
            print(f"[{idx_display}/{len(test_queries)}] ⚠️ SKIP (no question)")
            continue
        
        print(f"[{idx_display}/{len(test_queries)}] Q: {question[:60]}...")
        
        # Retrieve nodes for evaluation
        retrieved_nodes = query_engine.retrieve(question)
        
        # Extract retrieval_ids and retrieval_context
        retrieval_ids = []
        retrieval_context = []
        for source_node in retrieved_nodes:
            retrieval_ids.append(source_node.metadata.get('id', ''))
            retrieval_context.append(source_node.get_content())
        
        print(f"           Retrieved {len(retrieval_ids)} chunks")
        print(f"           Golden IDs: {golden_ids}")
        print(f"           Retrieval IDs: {retrieval_ids}")
        
        # Evaluate using TRT framework
        eval_result = evaluating_TRT(retrieval_ids, golden_ids)
        evaluateResults_TRT.add(eval_result)
        
        all_num += 1
        
        print(f"           Domain: {other_info.get('domain', 'medical')}")
        print(f"           Total queries: {all_num}")
        print(f"{'-'*60}")
        evaluateResults_TRT.print_results()
        print(f"{'-'*60}\n")
        
    except Exception as e:
        print(f"[{idx_display}/{len(test_queries)}] ❌ ERROR: {str(e)}")
        continue

print(f"\n{'='*60}")
print("EVALUATION COMPLETE")
print(f"{'='*60}")
print(f"Total Queries Processed: {all_num}/{len(test_queries)}")
print(f"\nFinal Results:")
evaluateResults_TRT.print_results()

# ========================================
# SAVE RESULTS
# ========================================

output_file = os.path.join(args.output_dir, f"medical_50_test_TRT_{model_name}.txt")
evaluateResults_TRT.print_results_to_path(output_file, cfg, model_name)
print(f"\n✅ Results saved to: {output_file}\n")
