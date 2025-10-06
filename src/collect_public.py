# src/collect_public.py
"""
Collects diverse queries from multiple public datasets for testing/evaluation.
Handles various dataset schemas and extracts normalized query strings.
"""
from datasets import load_dataset, get_dataset_config_names
import json
import random
import os
import logging
from typing import Optional, Dict, List
from pathlib import Path

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Constants
DATA_DIR = Path("data")
OUTPUT_FILE = DATA_DIR / "raw_queries.jsonl"
DEFAULT_SAMPLE_SIZE = 50
FINAL_SAMPLE_SIZE = 100
RANDOM_SEED = 42  # For reproducibility


def ensure_data_dir() -> None:
    """Create data directory if it doesn't exist."""
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    logger.info(f"Ensured data directory exists: {DATA_DIR}")


def safe_load(name: str, config: Optional[str] = None, split: str = f"train[:{DEFAULT_SAMPLE_SIZE}]"):
    """
    Load a dataset slice with automatic config fallback.
    
    Args:
        name: Dataset name on HuggingFace Hub
        config: Specific config/subset name (optional)
        split: Dataset split specification
        
    Returns:
        Dataset object
        
    Raises:
        Exception: If dataset cannot be loaded with any config
    """
    try:
        if config is None:
            logger.info(f"Loading {name} without config...")
            return load_dataset(name, split=split, trust_remote_code=True)
    except Exception as e:
        logger.warning(f"Failed to load {name} without config: {e}")
        
    # Try to find available configs
    try:
        cfgs = get_dataset_config_names(name)
        if not cfgs:
            logger.error(f"No configs available for {name}")
            raise ValueError(f"No configs available for dataset: {name}")
        
        config = config or cfgs[0]
        logger.info(f"Loading {name} with config '{config}'...")
        return load_dataset(name, config, split=split, trust_remote_code=True)
    
    except Exception as e:
        logger.error(f"Failed to load {name}: {e}")
        raise


def extract_query(dataset_name: str, example: Dict) -> Optional[str]:
    """
    Extract a query string from various dataset schemas.
    
    Args:
        dataset_name: Name of the source dataset
        example: Single example dictionary from the dataset
        
    Returns:
        Extracted query string or None if extraction fails
    """
    try:
        if dataset_name == "ms_marco":
            # MS MARCO v2.1 has 'query' field
            query = example.get("query")
            
        elif dataset_name == "hotpotqa":
            # HotpotQA uses 'question' field
            query = example.get("question")
            
        elif dataset_name == "adversarial_qa":
            # AdversarialQA follows SQuAD format with 'question'
            query = example.get("question")
            
        elif dataset_name == "realtoxicity":
            # RealToxicityPrompts has nested structure: 'prompt': {'text': '...'}
            prompt = example.get("prompt")
            query = prompt.get("text") if isinstance(prompt, dict) else None
            
        elif dataset_name == "daily_dialog":
            # DailyDialog has 'dialog' list; combine last 1-2 utterances
            dialog = example.get("dialog") or example.get("dialogue")
            if isinstance(dialog, list) and dialog:
                query = " ".join(dialog[-2:]) if len(dialog) > 1 else dialog[0]
            else:
                query = None
                
        else:
            # Generic fallback for unknown datasets
            query = (example.get("question") or 
                    example.get("query") or 
                    example.get("text"))
        
        # Validate and clean the query
        if query and isinstance(query, str):
            query = query.strip()
            return query if query else None
        return None
        
    except Exception as e:
        logger.warning(f"Error extracting query from {dataset_name}: {e}")
        return None


def load_datasets() -> Dict[str, any]:
    """
    Load all configured datasets.
    
    Returns:
        Dictionary mapping dataset names to loaded dataset objects
    """
    dataset_configs = {
        "ms_marco": ("microsoft/ms_marco", "v2.1"),
        "hotpotqa": ("hotpotqa/hotpot_qa", "distractor"),
        "adversarial_qa": ("adversarial_qa", "adversarialQA"),
        "realtoxicity": ("allenai/real-toxicity-prompts", None),
        "daily_dialog": ("daily_dialog", None),
    }
    
    datasets = {}
    for name, (hf_name, config) in dataset_configs.items():
        try:
            datasets[name] = safe_load(hf_name, config, split=f"train[:{DEFAULT_SAMPLE_SIZE}]")
            logger.info(f"✓ Loaded {name}")
        except Exception as e:
            logger.error(f"✗ Failed to load {name}: {e}")
            # Continue with other datasets
            
    return datasets


def collect_queries(datasets: Dict[str, any]) -> List[Dict[str, str]]:
    """
    Extract queries from all loaded datasets.
    
    Args:
        datasets: Dictionary of loaded dataset objects
        
    Returns:
        List of query dictionaries with 'query' and 'source' fields
    """
    raw_queries = []
    
    for dataset_name, dataset in datasets.items():
        logger.info(f"Processing {dataset_name}...")
        extracted_count = 0
        
        for example in dataset:
            query = extract_query(dataset_name, example)
            if query:
                raw_queries.append({
                    "query": query,
                    "source": dataset_name
                })
                extracted_count += 1
        
        logger.info(f"  Extracted {extracted_count} queries from {dataset_name}")
    
    return raw_queries


def sample_and_save(queries: List[Dict[str, str]], sample_size: int = FINAL_SAMPLE_SIZE) -> None:
    """
    Sample queries randomly and save to JSONL file.
    
    Args:
        queries: List of query dictionaries
        sample_size: Number of queries to sample
    """
    if not queries:
        logger.warning("No queries to save!")
        return
    
    # Shuffle with fixed seed for reproducibility
    random.seed(RANDOM_SEED)
    random.shuffle(queries)
    
    # Sample desired number
    sampled = queries[:sample_size]
    
    # Write to JSONL
    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        for query in sampled:
            f.write(json.dumps(query, ensure_ascii=False) + "\n")
    
    logger.info(f"✅ Saved {len(sampled)} queries to {OUTPUT_FILE}")
    
    # Print source distribution
    source_counts = {}
    for q in sampled:
        source_counts[q["source"]] = source_counts.get(q["source"], 0) + 1
    
    logger.info("Query distribution by source:")
    for source, count in sorted(source_counts.items()):
        logger.info(f"  {source}: {count}")


def main():
    """Main execution function."""
    logger.info("Starting query collection...")
    
    # Setup
    ensure_data_dir()
    
    # Load datasets
    datasets = load_datasets()
    
    if not datasets:
        logger.error("No datasets loaded successfully. Exiting.")
        return
    
    # Extract queries
    queries = collect_queries(datasets)
    
    logger.info(f"Total queries collected: {len(queries)}")
    
    # Sample and save
    sample_and_save(queries, FINAL_SAMPLE_SIZE)
    
    logger.info("Collection complete!")


if __name__ == "__main__":
    main()