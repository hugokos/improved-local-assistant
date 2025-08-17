#!/usr/bin/env python3
"""
Script to inspect the structure of prebuilt graph indices.

This script helps understand:
1. What indices are stored in each graph directory
2. The types and IDs of indices
3. The structure of index_store.json files
"""

import json
import logging
import sys
from pathlib import Path

# Add the project root to the path
sys.path.insert(0, str(Path(__file__).parent.parent))

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def inspect_graph_directory(graph_path: Path):
    """Inspect a single graph directory."""
    logger.info(f"\n=== Inspecting {graph_path.name} ===")

    if not graph_path.exists():
        logger.error(f"Directory does not exist: {graph_path}")
        return

    # List all files
    files = list(graph_path.glob("*"))
    logger.info(f"Files found: {[f.name for f in files]}")

    # Check for index_store.json
    index_store_path = graph_path / "index_store.json"
    if index_store_path.exists():
        try:
            with open(index_store_path, encoding="utf-8") as f:
                index_store = json.load(f)

            logger.info("Index store structure:")

            if "index_store/data" in index_store:
                indices = index_store["index_store/data"]
                logger.info(f"Found {len(indices)} indices:")

                for idx_id, idx_data in indices.items():
                    idx_type = idx_data.get("__type__", "unknown")
                    logger.info(f"  - ID: {idx_id}")
                    logger.info(f"    Type: {idx_type}")

                    # Parse the data if it's JSON
                    if "__data__" in idx_data:
                        try:
                            data = json.loads(idx_data["__data__"])
                            logger.info(f"    Index ID: {data.get('index_id', 'N/A')}")
                            logger.info(f"    Summary: {data.get('summary', 'N/A')}")

                            # For vector stores, show node count
                            if "nodes_dict" in data:
                                node_count = len(data["nodes_dict"])
                                logger.info(f"    Nodes: {node_count}")

                        except json.JSONDecodeError:
                            logger.warning(f"    Could not parse data for {idx_id}")
            else:
                logger.warning("No 'index_store/data' found in index_store.json")

        except Exception as e:
            logger.error(f"Error reading index_store.json: {e}")
    else:
        logger.warning("No index_store.json found")

    # Check for other important files
    important_files = [
        "docstore.json",
        "graph_store.json",
        "property_graph_store.json",
        "default__vector_store.json",
        "image__vector_store.json",
    ]

    for filename in important_files:
        file_path = graph_path / filename
        if file_path.exists():
            try:
                size = file_path.stat().st_size
                logger.info(f"✓ {filename}: {size:,} bytes")
            except Exception as e:
                logger.warning(f"✗ {filename}: Error reading - {e}")
        else:
            logger.info(f"- {filename}: Not found")


def test_loading_indices(graph_path: Path):
    """Test loading indices using LlamaIndex."""
    logger.info(f"\n=== Testing index loading for {graph_path.name} ===")

    try:
        from llama_index.core import PropertyGraphIndex
        from llama_index.core import Settings
        from llama_index.core import StorageContext
        from llama_index.core import VectorStoreIndex
        from llama_index.core import load_indices_from_storage
        from llama_index.embeddings.huggingface import HuggingFaceEmbedding
        from improved_local_assistant.services.utf8_import_helper import get_utf8_filesystem

        # Set up embedding model and disable LLM to avoid OpenAI error
        Settings.embed_model = HuggingFaceEmbedding(
            model_name="BAAI/bge-small-en-v1.5", device="cpu", normalize=True
        )
        Settings.llm = None  # Disable LLM for inspection

        # Load storage context
        storage_ctx = StorageContext.from_defaults(
            persist_dir=str(graph_path), fs=get_utf8_filesystem()
        )

        # Load all indices
        indices = load_indices_from_storage(storage_ctx)
        logger.info(f"Successfully loaded {len(indices)} indices")

        for i, idx in enumerate(indices):
            logger.info(f"Index {i+1}:")
            logger.info(f"  Type: {type(idx).__name__}")
            logger.info(f"  ID: {idx.index_id}")

            # Check if it's a vector or property graph index
            if isinstance(idx, VectorStoreIndex):
                logger.info("  ✓ VectorStoreIndex")
            elif isinstance(idx, PropertyGraphIndex):
                logger.info("  ✓ PropertyGraphIndex")
            else:
                logger.info("  ? Unknown index type")

            # Test retriever creation
            try:
                retriever = idx.as_retriever(similarity_top_k=3)
                logger.info(f"  ✓ Retriever created: {type(retriever).__name__}")
            except Exception as e:
                logger.warning(f"  ✗ Retriever creation failed: {e}")

    except Exception as e:
        logger.error(f"Failed to load indices: {e}")


def main():
    """Main inspection function."""
    logger.info("Starting graph indices inspection")

    # Path to prebuilt graphs
    graphs_path = Path("data/prebuilt_graphs")

    if not graphs_path.exists():
        logger.error(f"Prebuilt graphs directory not found: {graphs_path}")
        return False

    # Get all graph directories
    graph_dirs = [d for d in graphs_path.iterdir() if d.is_dir()]

    if not graph_dirs:
        logger.warning("No graph directories found")
        return False

    logger.info(f"Found {len(graph_dirs)} graph directories: {[d.name for d in graph_dirs]}")

    # Inspect each directory
    for graph_dir in graph_dirs:
        inspect_graph_directory(graph_dir)
        test_loading_indices(graph_dir)

    logger.info("\nInspection completed")
    return True


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
