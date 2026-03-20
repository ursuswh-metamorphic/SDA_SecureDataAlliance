"""
Generate synthetic training data for retriever from PubMed corpus.
Format: (query, positive_chunk, negative_chunks)

Process:
1. Load PubMed JSON
2. Chunk documents (sliding window)
3. Filter good chunks (length, quality)
4. Generate queries using LLM
5. Build (query, positive) pairs
6. Sample negatives from other chunks
7. Save dataset for FedE training
"""

import json
import random
import os
import sys
import logging
from typing import List, Dict, Tuple
import numpy as np
from collections import defaultdict

# Add parent paths for imports - ensure finsaferag package is importable
finsaferag_path = os.path.join(os.path.dirname(__file__), '..', 'finsaferag')
if finsaferag_path not in sys.path:
    sys.path.insert(0, finsaferag_path)

from finsaferag.llms.llm import get_llm
from finsaferag.config import Config

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class MedicalDatasetGenerator:
    def __init__(self, chunk_size: int = 256, chunk_overlap: int = 64, min_chunk_length: int = 100):
        """
        Initialize the dataset generator.
        
        Args:
            chunk_size: Tokens per chunk (approximate, split by sentences)
            chunk_overlap: Overlap between chunks
            min_chunk_length: Minimum chunk character length to keep
        """
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.min_chunk_length = min_chunk_length
        self.chunks: List[Dict] = []
        self.chunk_id_counter = 0
        
        # Initialize LLM
        cfg = Config()
        llm_name = getattr(cfg, "llm", "nvidia")
        logger.info(f"Using LLM: {llm_name}")
        self.llm = get_llm(llm_name)
    
    def load_pubmed(self, pubmed_path: str, max_docs: int = None) -> List[Dict]:
        """Load PubMed JSON file."""
        logger.info(f"Loading PubMed from {pubmed_path}")
        with open(pubmed_path, 'r', encoding='utf-8') as f:
            documents = json.load(f)
        
        if max_docs:
            documents = documents[:max_docs]
        
        logger.info(f"Loaded {len(documents)} documents")
        return documents
    
    def chunk_documents(self, documents: List[Dict]) -> List[Dict]:
        """
        Split documents into chunks using sliding window approach.
        """
        logger.info("Chunking documents...")
        chunks = []
        
        for doc in documents:
            # Use full content (title + content)
            text = doc.get('contents', doc.get('content', ''))
            if not text or len(text.strip()) < self.min_chunk_length:
                continue
            
            # Split by sentences (approximate)
            sentences = [s.strip() for s in text.split('.') if s.strip()]
            
            # Create overlapping chunks from sentences
            chunk_idx = 0
            while chunk_idx < len(sentences):
                chunk_sentences = sentences[chunk_idx:chunk_idx + self.chunk_size]
                chunk_text = '. '.join(chunk_sentences)
                
                if len(chunk_text) >= self.min_chunk_length:
                    chunk_obj = {
                        'id': f"{doc['id']}_chunk_{self.chunk_id_counter}",
                        'doc_id': doc['id'],
                        'title': doc.get('title', ''),
                        'text': chunk_text,
                        'pmid': doc.get('PMID', ''),
                        'start_sent': chunk_idx,
                        'num_sent': len(chunk_sentences)
                    }
                    chunks.append(chunk_obj)
                    self.chunk_id_counter += 1
                
                # Move by stride (chunk_size - overlap)
                chunk_idx += self.chunk_size - self.chunk_overlap
        
        logger.info(f"Created {len(chunks)} chunks")
        return chunks
    
    def filter_chunks(self, chunks: List[Dict]) -> List[Dict]:
        """
        Filter chunks based on quality metrics.
        Keep chunks that:
        - Have sufficient length
        - Contain medical/scientific terms
        """
        logger.info("Filtering chunks...")
        medical_keywords = [
            'patient', 'disease', 'treatment', 'therapy', 'drug', 'clinical',
            'study', 'method', 'result', 'analysis', 'research', 'data',
            'effect', 'symptom', 'diagnosis', 'medical', 'health', 'condition',
            'sample', 'test', 'rate', 'group', 'control', 'model', 'response'
        ]
        
        filtered = []
        for chunk in chunks:
            text_lower = chunk['text'].lower()
            
            # Check medical relevance
            medical_score = sum(1 for kw in medical_keywords if kw in text_lower)
            if medical_score < 2:
                continue
            
            # Check length
            if len(chunk['text']) < self.min_chunk_length:
                continue
            
            # Check for too much repetition
            words = text_lower.split()
            if len(words) > 0 and len(set(words)) / len(words) < 0.3:
                continue
            
            filtered.append(chunk)
        
        logger.info(f"Filtered to {len(filtered)} quality chunks")
        return filtered
    
    def generate_query_for_chunk(self, chunk_text: str, retry_count: int = 3) -> str:
        """
        Generate a query for a given chunk using LLM.
        """
        for attempt in range(retry_count):
            try:
                prompt = f"""Nhằm vào đoạn văn bản y tế sau, hãy tạo một câu hỏi tự nhiên mà có thể trả lời CHỈ dựa trên đoạn văn bản này. Câu hỏi nên:
- Chi tiết và cụ thể
- Không quá dài
- Khó trả lời mà không có đoạn văn bản

Đoạn văn bản:
{chunk_text[:800]}

Câu hỏi (chỉ output câu hỏi, không giải thích thêm):"""
                
                response = self.llm.complete(prompt)
                query = response.text.strip()
                
                # Clean up if LLM adds extra text
                if '\n' in query:
                    query = query.split('\n')[0]
                
                query = query.strip('"-\' ')
                
                if len(query) > 10 and len(query) < 500:
                    return query
                
            except Exception as e:
                logger.warning(f"Query generation attempt {attempt+1} failed: {e}")
                if attempt < retry_count - 1:
                    continue
        
        # Fallback: generate simple question
        words = chunk_text.split()[:20]
        return f"What does the text say about {' '.join(words[:5])}?"
    
    def build_training_pairs(self, chunks: List[Dict], queries_per_chunk: int = 1) -> List[Dict]:
        """
        Build training pairs: (query, positive_chunk, negative_chunks)
        """
        logger.info(f"Generating queries for {len(chunks)} chunks...")
        training_pairs = []
        
        # Generate queries and build pairs
        for i, chunk in enumerate(chunks):
            if (i + 1) % 10 == 0:
                logger.info(f"Processing chunk {i+1}/{len(chunks)}")
            
            for query_idx in range(queries_per_chunk):
                try:
                    # Generate query
                    query = self.generate_query_for_chunk(chunk['text'])
                    
                    # Sample negative chunks from different documents
                    negatives = self._sample_negatives(chunk, chunks, num_negatives=3)
                    
                    pair = {
                        'query': query,
                        'positive': {
                            'id': chunk['id'],
                            'doc_id': chunk['doc_id'],
                            'text': chunk['text'],
                            'title': chunk['title']
                        },
                        'negatives': negatives,
                        'metadata': {
                            'doc_id': chunk['doc_id'],
                            'pmid': chunk['pmid'],
                            'chunk_id': chunk['id']
                        }
                    }
                    training_pairs.append(pair)
                    
                except Exception as e:
                    logger.error(f"Failed to generate pair for chunk {chunk['id']}: {e}")
                    continue
        
        logger.info(f"Created {len(training_pairs)} training pairs")
        return training_pairs
    
    def _sample_negatives(self, positive_chunk: Dict, all_chunks: List[Dict], 
                         num_negatives: int = 3) -> List[Dict]:
        """
        Sample negative chunks from different documents.
        Strategy: sample random chunks from different doc_ids
        """
        candidates = [c for c in all_chunks 
                     if c['doc_id'] != positive_chunk['doc_id']]
        
        if len(candidates) < num_negatives:
            negatives = candidates
        else:
            negatives = random.sample(candidates, num_negatives)
        
        return [
            {
                'id': neg['id'],
                'doc_id': neg['doc_id'],
                'text': neg['text'],
                'title': neg['title']
            }
            for neg in negatives
        ]
    
    def save_dataset(self, training_pairs: List[Dict], output_path: str):
        """Save training dataset to JSON."""
        logger.info(f"Saving {len(training_pairs)} pairs to {output_path}")
        
        os.makedirs(os.path.dirname(output_path) or '.', exist_ok=True)
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(training_pairs, f, indent=2, ensure_ascii=False)
        
        logger.info(f"Dataset saved to {output_path}")
        
        # Print statistics
        self._print_statistics(training_pairs)
    
    def _print_statistics(self, training_pairs: List[Dict]):
        """Print dataset statistics."""
        if not training_pairs:
            return
        
        query_lengths = [len(p['query'].split()) for p in training_pairs]
        chunk_lengths = [len(p['positive']['text']) for p in training_pairs]
        
        logger.info("=" * 50)
        logger.info("DATASET STATISTICS")
        logger.info("=" * 50)
        logger.info(f"Total pairs: {len(training_pairs)}")
        logger.info(f"Query length - min: {min(query_lengths)}, max: {max(query_lengths)}, avg: {np.mean(query_lengths):.1f}")
        logger.info(f"Chunk length - min: {min(chunk_lengths)}, max: {max(chunk_lengths)}, avg: {np.mean(chunk_lengths):.1f}")
        logger.info(f"Negatives per pair: {len(training_pairs[0]['negatives'])}")
        logger.info("=" * 50)
    
    def generate_full_dataset(self, pubmed_path: str, output_path: str, 
                            max_docs: int = None, queries_per_chunk: int = 1):
        """
        Complete pipeline: load -> chunk -> filter -> generate -> save
        """
        # 1. Load PubMed
        documents = self.load_pubmed(pubmed_path, max_docs)
        
        # 2. Chunk documents
        chunks = self.chunk_documents(documents)
        
        # 3. Filter chunks
        chunks = self.filter_chunks(chunks)
        
        # 4. Build training pairs
        training_pairs = self.build_training_pairs(chunks, queries_per_chunk)
        
        # 5. Save dataset
        self.save_dataset(training_pairs, output_path)
        
        return training_pairs


def main():
    """Main execution."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Generate retriever training data from PubMed')
    parser.add_argument('--pubmed', type=str, 
                       default='pubmed23n0001.json',
                       help='Path to PubMed JSON file')
    parser.add_argument('--output', type=str,
                       default='pubmed_retriever_train.json',
                       help='Output dataset path')
    parser.add_argument('--max-docs', type=int, default=None,
                       help='Max documents to process')
    parser.add_argument('--queries-per-chunk', type=int, default=1,
                       help='Number of queries to generate per chunk')
    parser.add_argument('--chunk-size', type=int, default=256,
                       help='Chunk size in sentences')
    
    args = parser.parse_args()
    
    generator = MedicalDatasetGenerator(chunk_size=args.chunk_size)
    generator.generate_full_dataset(
        pubmed_path=args.pubmed,
        output_path=args.output,
        max_docs=args.max_docs,
        queries_per_chunk=args.queries_per_chunk
    )


if __name__ == "__main__":
    main()
