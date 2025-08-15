#!/usr/bin/env python3
"""
Test script to check HybridRetriever imports and availability.
"""

import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def test_hybrid_retriever_imports():
    """Test different HybridRetriever import paths."""

    # Test 1: llama_index.retrievers
    try:
        from llama_index.retrievers import HybridRetriever

        logger.info("✓ Successfully imported from llama_index.retrievers")
        logger.info(f"HybridRetriever: {HybridRetriever}")
        return HybridRetriever
    except ImportError as e:
        logger.warning(f"✗ Failed to import from llama_index.retrievers: {e}")

    # Test 2: llama_index.core.retrievers
    try:
        from llama_index.core.retrievers import HybridRetriever

        logger.info("✓ Successfully imported from llama_index.core.retrievers")
        logger.info(f"HybridRetriever: {HybridRetriever}")
        return HybridRetriever
    except ImportError as e:
        logger.warning(f"✗ Failed to import from llama_index.core.retrievers: {e}")

    # Test 3: llama_index.core.retrievers.fusion
    try:
        from llama_index.core.retrievers.fusion import HybridRetriever

        logger.info("✓ Successfully imported from llama_index.core.retrievers.fusion")
        logger.info(f"HybridRetriever: {HybridRetriever}")
        return HybridRetriever
    except ImportError as e:
        logger.warning(f"✗ Failed to import from llama_index.core.retrievers.fusion: {e}")

    # Test 4: Check what's available in core.retrievers
    try:
        import llama_index.core.retrievers as retrievers_module

        available = [
            attr for attr in dir(retrievers_module) if "Hybrid" in attr or "Fusion" in attr
        ]
        logger.info(f"Available hybrid/fusion retrievers in core.retrievers: {available}")
    except ImportError as e:
        logger.warning(f"Could not inspect core.retrievers: {e}")

    # Test 5: Check for QueryFusionRetriever as alternative
    try:
        from llama_index.core.retrievers import QueryFusionRetriever

        logger.info("✓ QueryFusionRetriever available as alternative")
        return QueryFusionRetriever
    except ImportError as e:
        logger.warning(f"✗ QueryFusionRetriever not available: {e}")

    logger.error("No hybrid retriever implementation found")
    return None


if __name__ == "__main__":
    hybrid_retriever_class = test_hybrid_retriever_imports()
    if hybrid_retriever_class:
        logger.info(f"Best available hybrid retriever: {hybrid_retriever_class.__name__}")
    else:
        logger.error("No hybrid retriever available")
