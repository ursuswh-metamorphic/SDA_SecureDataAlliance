"""
QUICK START SCRIPT - Generate retriever training data immediately

Chạy script này để nhanh chóng tạo training data với cấu hình mặc định.
"""

import os
import sys
import argparse
import logging

# Add paths - ensure finsaferag package is importable
finsaferag_path = os.path.join(os.path.dirname(__file__), '..', 'finsaferag')
if finsaferag_path not in sys.path:
    sys.path.insert(0, finsaferag_path)

from generate_retriever_training_data import MedicalDatasetGenerator
from dataset_utils import DatasetUtility

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def quick_start(preset: str = 'medium'):
    """
    Quick start with preset configurations.
    
    Presets:
      - tiny: 20 docs → ~100 pairs (test only)
      - small: 100 docs → ~500 pairs
      - medium: 500 docs → ~2500 pairs (recommended)
      - large: 1000 docs → ~5000 pairs
      - xl: all docs → ~50000+ pairs (slow!)
    """
    
    configs = {
        'tiny': {
            'max_docs': 20,
            'queries_per_chunk': 1,
            'output': 'pubmed_retriever_train_tiny.json',
            'description': 'Quick test (20 docs, ~100 pairs)'
        },
        'small': {
            'max_docs': 100,
            'queries_per_chunk': 1,
            'output': 'pubmed_retriever_train_small.json',
            'description': 'Small dataset (100 docs, ~500 pairs)'
        },
        'medium': {
            'max_docs': 500,
            'queries_per_chunk': 1,
            'output': 'pubmed_retriever_train_medium.json',
            'description': 'Medium dataset (500 docs, ~2500 pairs) - RECOMMENDED'
        },
        'large': {
            'max_docs': 1000,
            'queries_per_chunk': 1,
            'output': 'pubmed_retriever_train_large.json',
            'description': 'Large dataset (1000 docs, ~5000 pairs)'
        },
        'xl': {
            'max_docs': None,
            'queries_per_chunk': 1,
            'output': 'pubmed_retriever_train_full.json',
            'description': 'Full dataset (all docs, ~50000+ pairs) - SLOW!'
        }
    }
    
    if preset not in configs:
        logger.error(f"Unknown preset: {preset}")
        logger.info(f"Available: {', '.join(configs.keys())}")
        return False
    
    config = configs[preset]
    
    logger.info("\n" + "="*70)
    logger.info("QUICK START - RETRIEVER TRAINING DATA GENERATION")
    logger.info("="*70)
    logger.info(f"Preset: {preset}")
    logger.info(f"Description: {config['description']}")
    logger.info(f"Max docs: {config['max_docs'] or 'all'}")
    logger.info(f"Output: {config['output']}")
    logger.info("="*70 + "\n")
    
    # Check PubMed file exists
    pubmed_path = 'pubmed23n0001.json'
    if not os.path.exists(pubmed_path):
        logger.error(f"PubMed file not found: {pubmed_path}")
        return False
    
    # Generate dataset
    try:
        generator = MedicalDatasetGenerator()
        training_pairs = generator.generate_full_dataset(
            pubmed_path=pubmed_path,
            output_path=config['output'],
            max_docs=config['max_docs'],
            queries_per_chunk=config['queries_per_chunk']
        )
        
        logger.info("\n✓ Generation complete!")
        logger.info(f"Generated {len(training_pairs)} training pairs")
        logger.info(f"Saved to: {config['output']}")
        
        # Validate
        logger.info("\nValidating dataset...")
        is_valid, errors = DatasetUtility.validate_dataset(training_pairs)
        
        if is_valid:
            logger.info("✓ Dataset validation passed!")
            
            # Show statistics
            logger.info("\nAnalyzing dataset...")
            stats = DatasetUtility.analyze_dataset(training_pairs)
            DatasetUtility.print_statistics(stats)
        
        return True
        
    except Exception as e:
        logger.error(f"Error during generation: {e}", exc_info=True)
        return False


def split_for_federated(dataset_path: str, num_clients: int = 4):
    """Split generated dataset for federated clients."""
    logger.info(f"\nSplitting dataset for {num_clients} federated clients...")
    
    try:
        dataset = DatasetUtility.load_dataset(dataset_path)
        client_paths = DatasetUtility.split_dataset_for_clients(
            dataset, 
            num_clients=num_clients,
            output_dir='fede_client_datasets'
        )
        
        logger.info(f"✓ Dataset split into {num_clients} client datasets")
        logger.info(f"Output directory: fede_client_datasets/")
        
        # Show stats for each client
        logger.info("\nClient statistics:")
        for client_id, path in client_paths.items():
            client_data = DatasetUtility.load_dataset(path)
            logger.info(f"  Client {client_id}: {len(client_data)} pairs")
        
        return True
    
    except Exception as e:
        logger.error(f"Error splitting dataset: {e}", exc_info=True)
        return False


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='Quick start retriever training data generation'
    )
    parser.add_argument(
        '--preset', 
        choices=['tiny', 'small', 'medium', 'large', 'xl'],
        default='medium',
        help='Dataset size preset (default: medium)'
    )
    parser.add_argument(
        '--split-clients',
        type=int,
        default=None,
        help='Also split dataset for N federated clients'
    )
    parser.add_argument(
        '--list-presets',
        action='store_true',
        help='List available presets and exit'
    )
    
    args = parser.parse_args()
    
    # List presets
    if args.list_presets:
        logger.info("\nAvailable presets:")
        logger.info("  tiny    - 20 docs, ~100 pairs (quick test)")
        logger.info("  small   - 100 docs, ~500 pairs")
        logger.info("  medium  - 500 docs, ~2500 pairs (RECOMMENDED)")
        logger.info("  large   - 1000 docs, ~5000 pairs")
        logger.info("  xl      - all docs, ~50000+ pairs (SLOW!)")
        sys.exit(0)
    
    # Generate
    success = quick_start(args.preset)
    
    # Split if requested
    if success and args.split_clients:
        config_map = {
            'tiny': 'pubmed_retriever_train_tiny.json',
            'small': 'pubmed_retriever_train_small.json',
            'medium': 'pubmed_retriever_train_medium.json',
            'large': 'pubmed_retriever_train_large.json',
            'xl': 'pubmed_retriever_train_full.json',
        }
        dataset_path = config_map[args.preset]
        split_for_federated(dataset_path, args.split_clients)
    
    # Cleanup
    if not success:
        logger.error("\n✗ Generation failed. Check errors above.")
        sys.exit(1)
