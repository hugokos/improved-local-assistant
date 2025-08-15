#!/usr/bin/env python3
"""
Script to rebuild the survivalist knowledge graph from scratch.

This script deletes the existing survivalist graph and rebuilds it from the source files.
"""

import asyncio
import logging
import shutil
import sys
from pathlib import Path

# Add project root to Python path for imports
project_root = Path(__file__).resolve().parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Fix for Windows asyncio event loop issue
if sys.platform == "win32":
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())


async def rebuild_survivalist_graph():
    """Rebuild the survivalist knowledge graph from scratch."""
    from services.graph_manager import KnowledgeGraphManager
    from services.model_mgr import ModelConfig
    from services.model_mgr import ModelManager

    # Define paths
    source_file = Path(project_root) / "data" / "test_docs" / "sas_survival.txt"
    target_dir = Path(project_root) / "data" / "prebuilt_graphs" / "survivalist"

    # Check if source file exists
    if not source_file.exists():
        logger.error(f"Source file not found: {source_file}")
        return False

    # Delete existing graph directory if it exists
    if target_dir.exists():
        logger.info(f"Deleting existing graph directory: {target_dir}")
        try:
            shutil.rmtree(target_dir)
        except Exception as e:
            logger.error(f"Error deleting directory: {str(e)}")
            return False

    # Create target directory
    logger.info(f"Creating target directory: {target_dir}")
    target_dir.mkdir(parents=True, exist_ok=True)

    # Copy source file to target directory
    logger.info("Copying source file to target directory")
    with open(source_file, encoding="utf-8") as f:
        content = f.read()

    with open(target_dir / "sas_survival.txt", "w", encoding="utf-8") as f:
        f.write(content)

    # Initialize model manager
    logger.info("Initializing model manager")
    model_manager = ModelManager()

    # Get model configuration
    model_config = ModelConfig(
        name="hermes3:3b",
        type="conversation",
        context_window=8000,
        temperature=0.7,
        max_tokens=2048,
        timeout=120,
        max_parallel=2,
        max_loaded=2,
    )

    # Initialize model manager
    logger.info("Initializing models")
    success = await model_manager.initialize_models(model_config)

    if not success:
        logger.error("Failed to initialize models")
        return False

    # Initialize graph manager
    logger.info("Initializing graph manager")
    graph_manager = KnowledgeGraphManager(model_manager, {})

    # Create graph from documents
    logger.info("Creating graph from documents")
    graph_id = graph_manager.create_graph_from_documents(str(target_dir), "survivalist")

    if graph_id:
        logger.info(f"Successfully created graph with ID: {graph_id}")
        return True
    else:
        logger.error("Failed to create graph")
        return False


async def main():
    """Main function."""
    logger.info("Starting survivalist graph rebuild")

    if await rebuild_survivalist_graph():
        logger.info("Survivalist graph rebuild completed successfully")
        return 0
    else:
        logger.error("Survivalist graph rebuild failed")
        return 1


if __name__ == "__main__":
    asyncio.run(main())
