"""
Test script for retriever training data generation.
Generates a small dataset to verify the pipeline works correctly.
"""

import json
import sys
import os
import logging

# Add paths - ensure finsaferag package is importable
finsaferag_path = os.path.join(os.path.dirname(__file__), '..', 'finsaferag')
if finsaferag_path not in sys.path:
    sys.path.insert(0, finsaferag_path)

from generate_retriever_training_data import MedicalDatasetGenerator

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def test_small_dataset():
    """Generate a small test dataset with 5 documents."""
    logger.info("Starting small dataset test generation...")
    
    pubmed_path = 'pubmed23n0001.json'
    output_path = 'test_retriever_train_small.json'
    
    if not os.path.exists(pubmed_path):
        logger.error(f"PubMed file not found: {pubmed_path}")
        return
    
    try:
        generator = MedicalDatasetGenerator(chunk_size=256, chunk_overlap=64)
        
        # Generate with only 5 documents, 1 query per chunk
        training_pairs = generator.generate_full_dataset(
            pubmed_path=pubmed_path,
            output_path=output_path,
            max_docs=5,
            queries_per_chunk=1
        )
        
        logger.info(f"✓ Test dataset generated with {len(training_pairs)} pairs")
        
        # Display sample
        if training_pairs:
            logger.info("\n" + "="*60)
            logger.info("SAMPLE TRAINING PAIR")
            logger.info("="*60)
            sample = training_pairs[0]
            logger.info(f"Query: {sample['query']}")
            logger.info(f"\nPositive chunk (first 200 chars):")
            logger.info(sample['positive']['text'][:200] + "...")
            logger.info(f"\nNumber of negatives: {len(sample['negatives'])}")
            logger.info("="*60 + "\n")
        
        return output_path
        
    except Exception as e:
        logger.error(f"Error during test generation: {e}", exc_info=True)
        return None


def test_check_format():
    """Verify output format is correct."""
    logger.info("Checking dataset format...")
    
    output_path = 'test_retriever_train_small.json'
    
    if not os.path.exists(output_path):
        logger.error(f"Output file not found: {output_path}")
        return False
    
    try:
        with open(output_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        if not isinstance(data, list):
            logger.error("Dataset must be a list")
            return False
        
        if not data:
            logger.error("Dataset is empty")
            return False
        
        pair = data[0]
        required_keys = {'query', 'positive', 'negatives', 'metadata'}
        
        if not required_keys.issubset(pair.keys()):
            logger.error(f"Missing keys in pair. Required: {required_keys}, Got: {pair.keys()}")
            return False
        
        positive_keys = {'id', 'doc_id', 'text', 'title'}
        if not positive_keys.issubset(pair['positive'].keys()):
            logger.error(f"Missing keys in positive. Required: {positive_keys}")
            return False
        
        if not isinstance(pair['negatives'], list):
            logger.error("Negatives must be a list")
            return False
        
        if len(pair['negatives']) == 0:
            logger.error("Negatives list is empty")
            return False
        
        logger.info("✓ Format check passed!")
        logger.info(f"  - Sample has {len(data)} training pairs")
        logger.info(f"  - Each pair has {len(pair['negatives'])} negatives")
        return True
        
    except Exception as e:
        logger.error(f"Error checking format: {e}")
        return False


if __name__ == "__main__":
    logger.info("\n" + "="*60)
    logger.info("RETRIEVER TRAINING DATA GENERATION TEST")
    logger.info("="*60 + "\n")
    
    # Generate test data
    output = test_small_dataset()
    
    if output:
        # Check format
        logger.info()
        is_valid = test_check_format()
        
        if is_valid:
            logger.info("\n✓ All tests passed!")
        else:
            logger.error("\n✗ Format validation failed")
    else:
        logger.error("\n✗ Test generation failed")
