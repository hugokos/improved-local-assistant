#!/usr/bin/env python3
"""
Script to rebuild knowledge graphs with local embeddings.

This script rebuilds all existing knowledge graphs using the local embedding model
to avoid mixed-dimensionality errors when switching from OpenAI embeddings.
"""

import os
import sys
import asyncio
import logging
import yaml

# Add parent directory to path to import from services
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from services.model_mgr import ModelManager, ModelConfig
from services.graph_manager import KnowledgeGraphManager

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Fix for Windows asyncio event loop issue
if sys.platform == "win32":
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

def load_config():
    """Load configuration from config.yaml."""
    config_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "config.yaml")
    
    if os.path.exists(config_path):
        with open(config_path, 'r') as f:
            return yaml.safe_load(f)
    else:
        logger.warning(f"Config file not found at {config_path}, using default configuration")
        return {
            "models": {
                "conversation": {
                    "name": "hermes3:3b",
                    "context_window": 8000,
                    "temperature": 0.7,
                    "max_tokens": 2048
                },
                "knowledge": {
                    "name": "tinyllama",
                    "context_window": 2048,
                    "temperature": 0.2,
                    "max_tokens": 1024
                }
            },
            "conversation": {
                "max_history_length": 50,
                "summarize_threshold": 20
            },
            "knowledge_graphs": {
                "prebuilt_directory": "./data/prebuilt_graphs",
                "dynamic_storage": "./data/dynamic_graph",
                "max_triplets_per_chunk": 4
            },
            "ollama": {
                "host": "http://localhost:11434",
                "timeout": 120
            }
        }

async def main():
    """Main function to rebuild knowledge graphs."""
    # Load configuration
    config = load_config()
    
    # Initialize model manager
    logger.info("Initializing model manager...")
    model_manager = ModelManager(host=config["ollama"]["host"])
    
    # Create model configuration
    model_config = ModelConfig(
        name=config["models"]["conversation"]["name"],
        type="conversation",
        context_window=config["models"]["conversation"]["context_window"],
        temperature=config["models"]["conversation"]["temperature"],
        max_tokens=config["models"]["conversation"]["max_tokens"],
        timeout=config["ollama"]["timeout"],
        max_parallel=2,
        max_loaded=2
    )
    
    # Initialize models
    logger.info("Initializing models...")
    if not await model_manager.initialize_models(model_config):
        logger.error("Failed to initialize models")
        return
    
    # Initialize knowledge graph manager
    logger.info("Initializing knowledge graph manager...")
    kg_manager = KnowledgeGraphManager(model_manager, config)
    
    # Load pre-built graphs if available
    logger.info("Loading pre-built knowledge graphs...")
    
    # Ensure proper encoding for Windows
    if sys.platform == "win32":
        logger.info("Setting console to UTF-8 mode for Windows")
        try:
            import subprocess
            subprocess.run(["chcp", "65001"], shell=True, check=False)
        except Exception as e:
            logger.warning(f"Failed to set console code page: {str(e)}")
    
    # Set default encoding to UTF-8
    import importlib
    try:
        importlib.reload(sys)
        sys.setdefaultencoding('utf-8')
    except (AttributeError, NameError):
        # Python 3 doesn't need this and doesn't have setdefaultencoding
        pass
    
    loaded_graphs = kg_manager.load_prebuilt_graphs()
    if loaded_graphs:
        logger.info(f"Loaded {len(loaded_graphs)} pre-built knowledge graphs")
    else:
        logger.info("No pre-built knowledge graphs loaded")
    
    # Initialize dynamic graph
    logger.info("Initializing dynamic knowledge graph...")
    kg_manager.initialize_dynamic_graph()
    
    # Rebuild all graphs with new embeddings
    logger.info("Rebuilding all knowledge graphs with local embeddings...")
    results = kg_manager.rebuild_graphs_with_new_embeddings()
    
    # Print results
    logger.info("Rebuild results:")
    for graph_id, success in results.items():
        status = "✓ Success" if success else "✗ Failed"
        logger.info(f"  {graph_id}: {status}")
    
    # Get graph statistics
    stats = kg_manager.get_graph_statistics()
    logger.info(f"Total graphs: {stats['total_graphs']}")
    logger.info(f"Total nodes: {stats['total_nodes']}")
    logger.info(f"Total edges: {stats['total_edges']}")
    
    logger.info("Knowledge graph rebuild complete")

if __name__ == "__main__":
    asyncio.run(main())