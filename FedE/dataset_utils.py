"""
Utility functions for retriever training dataset management.
- Split dataset for federated clients
- Analyze dataset statistics
- Validate dataset integrity
"""

import json
import os
import logging
from typing import List, Dict, Tuple
from pathlib import Path

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DatasetUtility:
    """Utilities for managing retriever training datasets."""
    
    @staticmethod
    def load_dataset(path: str) -> List[Dict]:
        """Load training dataset from JSON."""
        logger.info(f"Loading dataset from {path}")
        with open(path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        logger.info(f"Loaded {len(data)} training pairs")
        return data
    
    @staticmethod
    def save_dataset(data: List[Dict], path: str):
        """Save dataset to JSON."""
        logger.info(f"Saving {len(data)} pairs to {path}")
        os.makedirs(os.path.dirname(path) or '.', exist_ok=True)
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        logger.info(f"Saved to {path}")
    
    @staticmethod
    def split_dataset_for_clients(dataset: List[Dict], num_clients: int, 
                                  output_dir: str = "client_data") -> Dict[int, str]:
        """
        Split dataset evenly across federated clients.
        
        Args:
            dataset: Full training dataset
            num_clients: Number of clients to split to
            output_dir: Output directory for client datasets
        
        Returns:
            Dict mapping client_id to dataset path
        """
        logger.info(f"Splitting {len(dataset)} pairs across {num_clients} clients...")
        
        os.makedirs(output_dir, exist_ok=True)
        chunk_size = len(dataset) // num_clients
        
        client_paths = {}
        for client_id in range(num_clients):
            start_idx = client_id * chunk_size
            if client_id == num_clients - 1:
                # Last client gets remaining data
                end_idx = len(dataset)
            else:
                end_idx = (client_id + 1) * chunk_size
            
            client_data = dataset[start_idx:end_idx]
            client_path = os.path.join(output_dir, f"client_{client_id}_data.json")
            
            DatasetUtility.save_dataset(client_data, client_path)
            client_paths[client_id] = client_path
            
            logger.info(f"  Client {client_id}: {len(client_data)} pairs → {client_path}")
        
        return client_paths
    
    @staticmethod
    def analyze_dataset(dataset: List[Dict]) -> Dict:
        """Compute dataset statistics."""
        if not dataset:
            return {}
        
        stats = {
            'total_pairs': len(dataset),
            'query_stats': {},
            'positive_stats': {},
            'negative_stats': {},
            'doc_coverage': {}
        }
        
        query_lengths = []
        positive_lengths = []
        negative_counts = []
        doc_ids = set()
        
        for pair in dataset:
            # Query stats
            query = pair.get('query', '')
            query_lengths.append(len(query.split()))
            
            # Positive stats
            positive_text = pair.get('positive', {}).get('text', '')
            positive_lengths.append(len(positive_text))
            
            # Negative stats
            negatives = pair.get('negatives', [])
            negative_counts.append(len(negatives))
            
            # Doc coverage
            doc_id = pair.get('metadata', {}).get('doc_id', 'unknown')
            doc_ids.add(doc_id)
        
        stats['query_stats'] = {
            'min': min(query_lengths),
            'max': max(query_lengths),
            'mean': sum(query_lengths) / len(query_lengths),
            'median': sorted(query_lengths)[len(query_lengths)//2]
        }
        
        stats['positive_stats'] = {
            'min': min(positive_lengths),
            'max': max(positive_lengths),
            'mean': sum(positive_lengths) / len(positive_lengths),
            'median': sorted(positive_lengths)[len(positive_lengths)//2]
        }
        
        stats['negative_stats'] = {
            'min': min(negative_counts),
            'max': max(negative_counts),
            'mean': sum(negative_counts) / len(negative_counts),
        }
        
        stats['doc_coverage'] = {
            'unique_docs': len(doc_ids),
            'pairs_per_doc_mean': len(dataset) / len(doc_ids) if doc_ids else 0
        }
        
        return stats
    
    @staticmethod
    def print_statistics(stats: Dict):
        """Print formatted statistics."""
        logger.info("\n" + "="*60)
        logger.info("DATASET STATISTICS")
        logger.info("="*60)
        
        logger.info(f"Total pairs: {stats.get('total_pairs', 0)}")
        
        logger.info("\nQuery Statistics (words):")
        qs = stats.get('query_stats', {})
        logger.info(f"  Min: {qs.get('min', 0)}, Max: {qs.get('max', 0)}, "
                   f"Mean: {qs.get('mean', 0):.1f}, Median: {qs.get('median', 0)}")
        
        logger.info("\nPositive Chunk Statistics (characters):")
        ps = stats.get('positive_stats', {})
        logger.info(f"  Min: {ps.get('min', 0)}, Max: {ps.get('max', 0)}, "
                   f"Mean: {ps.get('mean', 0):.1f}, Median: {ps.get('median', 0)}")
        
        logger.info("\nNegatives per Pair:")
        ns = stats.get('negative_stats', {})
        logger.info(f"  Min: {ns.get('min', 0)}, Max: {ns.get('max', 0)}, "
                   f"Mean: {ns.get('mean', 0):.1f}")
        
        logger.info("\nDocument Coverage:")
        dc = stats.get('doc_coverage', {})
        logger.info(f"  Unique docs: {dc.get('unique_docs', 0)}")
        logger.info(f"  Pairs per doc (mean): {dc.get('pairs_per_doc_mean', 0):.1f}")
        
        logger.info("="*60 + "\n")
    
    @staticmethod
    def validate_dataset(dataset: List[Dict]) -> Tuple[bool, List[str]]:
        """Validate dataset integrity and format."""
        errors = []
        
        if not isinstance(dataset, list):
            errors.append("Dataset must be a list")
            return False, errors
        
        if not dataset:
            errors.append("Dataset is empty")
            return False, errors
        
        # Check first few pairs
        for idx, pair in enumerate(dataset[:min(10, len(dataset))]):
            # Required keys
            if 'query' not in pair:
                errors.append(f"Pair {idx}: missing 'query'")
            if 'positive' not in pair:
                errors.append(f"Pair {idx}: missing 'positive'")
            if 'negatives' not in pair:
                errors.append(f"Pair {idx}: missing 'negatives'")
            
            # Check positive structure
            if 'positive' in pair:
                positive = pair['positive']
                required = {'id', 'doc_id', 'text', 'title'}
                if not required.issubset(positive.keys()):
                    errors.append(f"Pair {idx}: positive missing keys {required - set(positive.keys())}")
            
            # Check negatives
            if 'negatives' in pair:
                negatives = pair['negatives']
                if not isinstance(negatives, list):
                    errors.append(f"Pair {idx}: 'negatives' must be a list")
                elif len(negatives) == 0:
                    errors.append(f"Pair {idx}: 'negatives' is empty")
                else:
                    # Check negative structure
                    for neg_idx, neg in enumerate(negatives):
                        if 'text' not in neg:
                            errors.append(f"Pair {idx}, negative {neg_idx}: missing 'text'")
        
        is_valid = len(errors) == 0
        if is_valid:
            logger.info("✓ Dataset validation passed!")
        else:
            logger.error(f"✗ Dataset validation found {len(errors)} errors:")
            for error in errors[:10]:  # Show first 10
                logger.error(f"  - {error}")
            if len(errors) > 10:
                logger.error(f"  ... and {len(errors)-10} more errors")
        
        return is_valid, errors
    
    @staticmethod
    def merge_datasets(dataset_paths: List[str], output_path: str):
        """Merge multiple datasets into one."""
        logger.info(f"Merging {len(dataset_paths)} datasets...")
        
        merged = []
        for path in dataset_paths:
            data = DatasetUtility.load_dataset(path)
            merged.extend(data)
        
        DatasetUtility.save_dataset(merged, output_path)
        logger.info(f"Merged dataset: {len(merged)} pairs")
        
        return merged
    
    @staticmethod
    def sample_dataset(dataset: List[Dict], sample_size: int, 
                       output_path: str = None) -> List[Dict]:
        """Random sample from dataset."""
        import random
        
        logger.info(f"Sampling {sample_size} pairs from {len(dataset)} pairs")
        sampled = random.sample(dataset, min(sample_size, len(dataset)))
        
        if output_path:
            DatasetUtility.save_dataset(sampled, output_path)
        
        return sampled


def main_split_example():
    """Example: Split dataset for 4 federated clients."""
    # Load dataset
    dataset = DatasetUtility.load_dataset('pubmed_retriever_train.json')
    
    # Split for 4 clients
    client_paths = DatasetUtility.split_dataset_for_clients(
        dataset, 
        num_clients=4,
        output_dir='fede_client_datasets'
    )
    
    # Analyze each client's data
    for client_id, path in client_paths.items():
        client_data = DatasetUtility.load_dataset(path)
        stats = DatasetUtility.analyze_dataset(client_data)
        logger.info(f"\nClient {client_id}:")
        DatasetUtility.print_statistics(stats)


def main_validate_example():
    """Example: Validate a dataset."""
    dataset = DatasetUtility.load_dataset('pubmed_retriever_train.json')
    
    is_valid, errors = DatasetUtility.validate_dataset(dataset)
    
    if is_valid:
        stats = DatasetUtility.analyze_dataset(dataset)
        DatasetUtility.print_statistics(stats)
    else:
        logger.error(f"Dataset has {len(errors)} validation errors")


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) < 2:
        logger.info("Usage:")
        logger.info("  python dataset_utils.py split <dataset.json> <num_clients>")
        logger.info("  python dataset_utils.py validate <dataset.json>")
        logger.info("  python dataset_utils.py analyze <dataset.json>")
    
    elif sys.argv[1] == 'split':
        dataset_path = sys.argv[2] if len(sys.argv) > 2 else 'pubmed_retriever_train.json'
        num_clients = int(sys.argv[3]) if len(sys.argv) > 3 else 4
        
        dataset = DatasetUtility.load_dataset(dataset_path)
        DatasetUtility.split_dataset_for_clients(dataset, num_clients)
    
    elif sys.argv[1] == 'validate':
        dataset_path = sys.argv[2] if len(sys.argv) > 2 else 'pubmed_retriever_train.json'
        dataset = DatasetUtility.load_dataset(dataset_path)
        is_valid, errors = DatasetUtility.validate_dataset(dataset)
        
        if is_valid:
            stats = DatasetUtility.analyze_dataset(dataset)
            DatasetUtility.print_statistics(stats)
    
    elif sys.argv[1] == 'analyze':
        dataset_path = sys.argv[2] if len(sys.argv) > 2 else 'pubmed_retriever_train.json'
        dataset = DatasetUtility.load_dataset(dataset_path)
        stats = DatasetUtility.analyze_dataset(dataset)
        DatasetUtility.print_statistics(stats)
