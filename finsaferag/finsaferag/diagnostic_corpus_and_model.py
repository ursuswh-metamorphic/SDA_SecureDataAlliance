"""
Comprehensive Corpus & Embedding Model Quality Diagnostic
==========================================================
Checks:
1. Corpus data quality (duplicates, empty content, text stats)
2. Embedding model quality (similarity distribution, clustering)
3. Model training effectiveness
"""

import json
import numpy as np
from collections import Counter
import hashlib
from sentence_transformers import SentenceTransformer
import torch

def hash_text(text):
    """Generate hash of text for duplicate detection"""
    return hashlib.md5(text.lower().strip().encode()).hexdigest()

def check_corpus_quality(corpus_path):
    """Analyze corpus document quality"""
    print("\n" + "="*70)
    print("1. CORPUS QUALITY CHECK")
    print("="*70)
    
    with open(corpus_path, 'r') as f:
        corpus = json.load(f)
    
    print(f"\nTotal documents: {len(corpus)}")
    
    # Check for empty/short documents
    text_lengths = []
    empty_docs = 0
    short_docs = 0
    
    for doc in corpus:
        text = doc.get('text', '')
        if not text or len(text.strip()) == 0:
            empty_docs += 1
        text_lengths.append(len(text))
        if len(text) < 100:
            short_docs += 1
    
    print(f"\nText Length Statistics:")
    print(f"  Empty documents: {empty_docs}")
    print(f"  Too short (<100 chars): {short_docs}")
    print(f"  Min length: {min(text_lengths)} chars")
    print(f"  Max length: {max(text_lengths)} chars")
    print(f"  Avg length: {np.mean(text_lengths):.0f} chars")
    print(f"  Median length: {np.median(text_lengths):.0f} chars")
    
    # Check for duplicates
    hashes = []
    for doc in corpus:
        text = doc.get('text', '')
        h = hash_text(text)
        hashes.append(h)
    
    unique_hashes = len(set(hashes))
    duplicates = len(hashes) - unique_hashes
    
    print(f"\nDuplicate Detection:")
    print(f"  Unique documents: {unique_hashes}")
    print(f"  Duplicate documents: {duplicates}")
    print(f"  Duplication rate: {100*duplicates/len(corpus):.2f}%")
    
    if duplicates > 0:
        hash_counts = Counter(hashes)
        most_common = hash_counts.most_common(3)
        print(f"  Most duplicated (top 3):")
        for h, count in most_common:
            if count > 1:
                # Find document with this hash
                for doc in corpus:
                    if hash_text(doc.get('text', '')) == h:
                        preview = doc.get('text', '')[:80]
                        print(f"    - Appears {count}x: {preview}...")
                        break
    
    # Check for common issues
    print(f"\nCommon Content Issues:")
    issue_count = 0
    for i, doc in enumerate(corpus[:100]):  # Check first 100
        text = doc.get('text', '')
        if text.count('\n') > 50:
            print(f"  Doc {i}: Has {text.count(chr(10))} newlines (possible formatting issue)")
            issue_count += 1
        if text.count('  ') > len(text) / 20:
            print(f"  Doc {i}: Excessive whitespace")
            issue_count += 1
    
    if issue_count == 0:
        print("  ✓ No obvious formatting issues detected")
    
    return corpus

def check_embedding_model(model_path, corpus):
    """Analyze embedding model quality"""
    print("\n" + "="*70)
    print("2. EMBEDDING MODEL QUALITY CHECK")
    print("="*70)
    
    print(f"\nLoading model from: {model_path}")
    try:
        # Try loading as SentenceTransformer
        model = SentenceTransformer(model_path)
        print(f"✓ Model loaded successfully")
        print(f"  Model type: SentenceTransformer")
        print(f"  Device: {model.device}")
    except Exception as e:
        print(f"✗ Failed to load model: {e}")
        return None
    
    # Get model info
    try:
        print(f"  Embedding dimension: {model.get_sentence_embedding_dimension()}")
    except:
        pass
    
    # Sample embeddings from corpus
    print(f"\nSampling {min(100, len(corpus))} documents for embedding analysis...")
    sample_size = min(100, len(corpus))
    sample_docs = corpus[:sample_size]
    
    texts = [doc.get('text', '')[:512] for doc in sample_docs]  # First 512 chars
    
    try:
        embeddings = model.encode(texts, show_progress_bar=False)
        print(f"✓ Embeddings generated: shape {embeddings.shape}")
    except Exception as e:
        print(f"✗ Failed to generate embeddings: {e}")
        return None
    
    # Analyze embedding quality
    print(f"\nEmbedding Statistics:")
    print(f"  Norm (L2): min={np.min(np.linalg.norm(embeddings, axis=1)):.4f}, "
          f"max={np.max(np.linalg.norm(embeddings, axis=1)):.4f}, "
          f"mean={np.mean(np.linalg.norm(embeddings, axis=1)):.4f}")
    
    # Compute similarity matrix
    print(f"\nComputing document similarities (this may take a moment)...")
    similarities = []
    for i in range(len(embeddings)):
        for j in range(i+1, len(embeddings)):
            sim = np.dot(embeddings[i], embeddings[j]) / (
                np.linalg.norm(embeddings[i]) * np.linalg.norm(embeddings[j]) + 1e-8
            )
            similarities.append(sim)
    
    similarities = np.array(similarities)
    print(f"\nSimilarity Distribution (cosine):")
    print(f"  Min: {np.min(similarities):.4f}")
    print(f"  Max: {np.max(similarities):.4f}")
    print(f"  Mean: {np.mean(similarities):.4f}")
    print(f"  Median: {np.median(similarities):.4f}")
    print(f"  Std: {np.std(similarities):.4f}")
    
    # Check distribution quality
    high_sim_ratio = (similarities > 0.8).sum() / len(similarities) * 100
    low_sim_ratio = (similarities < 0.2).sum() / len(similarities) * 100
    
    print(f"\nSimilarity Range Analysis:")
    print(f"  >0.8 (very similar): {high_sim_ratio:.1f}%")
    print(f"  0.2-0.8 (medium): {100 - high_sim_ratio - low_sim_ratio:.1f}%")
    print(f"  <0.2 (dissimilar): {low_sim_ratio:.1f}%")
    
    # Quality assessment
    print(f"\nModel Quality Assessment:")
    if high_sim_ratio > 50:
        print(f"  ⚠️ WARNING: Too many similar documents ({high_sim_ratio:.1f}%)")
        print(f"     → Model may lack diversity or corpus has duplicates")
    elif high_sim_ratio < 5:
        print(f"  ⚠️ WARNING: Too few similar documents ({high_sim_ratio:.1f}%)")
        print(f"     → Model may be undertrained or corpus too diverse")
    else:
        print(f"  ✓ Good similarity distribution")
    
    # Check isotropy (directions well-distributed)
    print(f"\nIsotropy Check (direction distribution):")
    # Normalize embeddings
    norm_emb = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)
    # Compute pairwise cosine similarities
    cosine_sims = np.dot(norm_emb, norm_emb.T)
    # Remove diagonal (self-similarity = 1)
    avg_cosine_sim = np.mean(cosine_sims[~np.eye(len(cosine_sims), dtype=bool)])
    print(f"  Average off-diagonal cosine sim: {avg_cosine_sim:.4f}")
    if avg_cosine_sim > 0.5:
        print(f"  ⚠️ WARNING: Low isotropy ({avg_cosine_sim:.4f} is high)")
        print(f"     → Embeddings may be collapsed into few directions")
        print(f"     → Model may need more training or better data")
    else:
        print(f"  ✓ Good isotropy")
    
    return model, embeddings

def check_model_training_quality(model_path, corpus, embeddings=None):
    """Check if model training was effective"""
    print("\n" + "="*70)
    print("3. MODEL TRAINING QUALITY CHECK")
    print("="*70)
    
    print(f"\nModel path: {model_path}")
    
    # Load model config if available
    import os
    config_path = os.path.join(model_path, 'config.json')
    
    if os.path.exists(config_path):
        print(f"✓ Found model config")
        with open(config_path, 'r') as f:
            config = json.load(f)
            if 'model_type' in config:
                print(f"  Model type: {config.get('model_type')}")
            if 'vocab_size' in config:
                print(f"  Vocab size: {config.get('vocab_size')}")
            if 'hidden_size' in config:
                print(f"  Hidden size: {config.get('hidden_size')}")
    
    # Check for sentence-transformers training info
    try:
        model = SentenceTransformer(model_path)
        if hasattr(model, 'get_sentence_embedding_dimension'):
            dim = model.get_sentence_embedding_dimension()
            print(f"  Embedding dimension: {dim}")
            if dim < 256:
                print(f"  ⚠️ WARNING: Small embedding dimension ({dim})")
                print(f"     → May limit model capacity for complex tasks")
    except:
        pass
    
    # Training effectiveness indicators
    print(f"\nTraining Quality Indicators:")
    
    norm_stability = None
    condition_number = None
    effective_rank = None
    emb_dim = None
    
    if embeddings is not None:
        # 1. Check normalized embedding magnitude (should be close to 1)
        norms = np.linalg.norm(embeddings, axis=1)
        norm_stability = np.std(norms) / np.mean(norms)
        print(f"  Embedding norm stability: {norm_stability:.4f}")
        if norm_stability > 0.3:
            print(f"    ⚠️ High variance in embedding norms - possible training issue")
        
        # 2. Check eigenvalues of embedding covariance
        cov = np.cov(embeddings.T)
        eigenvalues = np.linalg.eigvalsh(cov)
        condition_number = eigenvalues[-1] / (eigenvalues[0] + 1e-8)
        print(f"  Covariance condition number: {condition_number:.2f}")
        if condition_number > 100:
            print(f"    ⚠️ High condition number - embeddings may be poorly trained")
        else:
            print(f"    ✓ Good conditioning")
        
        # 3. Effective rank
        total_variance = np.sum(eigenvalues)
        cumsum = np.cumsum(eigenvalues[::-1])
        effective_rank = np.argmax(cumsum >= 0.9 * total_variance) + 1
        emb_dim = embeddings.shape[1]
        print(f"  Effective rank (90% variance): {effective_rank}")
        if effective_rank < emb_dim * 0.3:
            print(f"    ⚠️ Very low effective rank ({effective_rank}/{emb_dim})")
            print(f"       → Embeddings may be degenerate or undertrained")
        else:
            print(f"    ✓ Good effective rank utilization")
    
    # 4. Summary diagnosis
    print(f"\n{'='*70}")
    print("SUMMARY DIAGNOSIS")
    print(f"{'='*70}")
    
    issues = []
    
    # Read corpus for analysis
    with open('/root/SDA_SecureDataAlliance/finsaferag/finsaferag/data/rag_corpus.json', 'r') as f:
        corpus_check = json.load(f)
    
    # Check corpus
    hashes = [hash_text(doc.get('text', '')) for doc in corpus_check]
    if len(set(hashes)) < len(hashes) * 0.95:
        issues.append("❌ Corpus has significant duplicates (>5%)")
    
    empty = sum(1 for doc in corpus_check if len(doc.get('text', '')) < 50)
    if empty > len(corpus_check) * 0.05:
        issues.append("❌ Corpus has too many short/empty documents (>5%)")
    
    # Print recommendations
    if issues:
        print("\n🔴 ISSUES FOUND:")
        for issue in issues:
            print(f"  {issue}")
    else:
        print("\n🟢 Corpus appears clean")
    
    print("\n📋 RECOMMENDATIONS:")
    if norm_stability is not None and norm_stability > 0.3:
        print("  1. Retrain embedding model with better normalization")
    
    if condition_number is not None and condition_number > 100:
        print("  2. Model may be undertrained:")
        print("     - Increase training epochs")
        print("     - Check training data quality")
        print("     - Verify loss function is decreasing")
    
    print("  3. Consider using pretrained BGE model if custom training is ineffective")

if __name__ == "__main__":
    corpus_path = '/root/SDA_SecureDataAlliance/finsaferag/finsaferag/data/rag_corpus.json'
    model_path = '/root/SDA_SecureDataAlliance/finsaferag/finsaferag/x-model_2026-03-19_13-06-32_converted'
    
    # Run diagnostics
    corpus = check_corpus_quality(corpus_path)
    result = check_embedding_model(model_path, corpus)
    
    if result:
        model, embeddings = result
        check_model_training_quality(model_path, corpus, embeddings)
    else:
        print("\n⚠️ Could not complete model training check")
