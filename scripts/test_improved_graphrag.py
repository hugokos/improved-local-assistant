#!/usr/bin/env python3
"""
Test script for the improved GraphRAG system with multiple indices support.

This script tests:
1. Loading multiple indices from prebuilt graphs
2. Creating hybrid retrievers combining vector and property graph search
3. Querying with the improved system
"""

import asyncio
import logging
import sys
from pathlib import Path

# Add the project root to the path
sys.path.insert(0, str(Path(__file__).parent.parent))

from services.embedding_singleton import configure_global_embedding
from services.embedding_singleton import get_embedding_model
from services.graph_router import GraphRouter

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


async def test_improved_graphrag():
    """Test the improved GraphRAG system."""
    try:
        logger.info("Starting improved GraphRAG test")

        # Initialize embedding model
        embedding_model = get_embedding_model("BAAI/bge-small-en-v1.5")
        configure_global_embedding("BAAI/bge-small-en-v1.5")

        # Configuration for GraphRouter
        config = {
            "embedding_model": "BAAI/bge-small-en-v1.5",
            "registry_path": "data/graph_registry.json",
            "data_path": "data/prebuilt_graphs",
            "max_cached_indices": 10,
            "hybrid_alpha": 0.6,
            "rerank_top_n": 10,
            "min_score_threshold": 0.4,
        }

        # Initialize GraphRouter with embedding model
        router = GraphRouter(config, embedder=embedding_model)

        # Initialize the router
        success = await router.initialize()
        if not success:
            logger.error("Failed to initialize GraphRouter")
            return False

        logger.info("GraphRouter initialized successfully")

        # Test loading multiple indices from survivalist graph
        logger.info("Testing multiple indices loading...")
        vector_idx, property_idx = router.get_indices("survivalist")

        if vector_idx:
            logger.info(
                f"✓ Vector index loaded: {type(vector_idx).__name__} (ID: {vector_idx.index_id})"
            )
        else:
            logger.warning("✗ No vector index found")

        if property_idx:
            logger.info(
                f"✓ Property index loaded: {type(property_idx).__name__} (ID: {property_idx.index_id})"
            )
        else:
            logger.warning("✗ No property index found")

        # Test hybrid retriever creation
        logger.info("Testing hybrid retriever creation...")
        retriever = router.get_retriever("survivalist")

        if retriever:
            logger.info(f"✓ Hybrid retriever created: {type(retriever).__name__}")
        else:
            logger.error("✗ Failed to create hybrid retriever")
            return False

        # Test queries
        test_queries = [
            "How do I start a fire in survival situations?",
            "What are the best water purification methods?",
            "How to build a shelter in the wilderness?",
            "What plants are safe to eat in survival?",
        ]

        logger.info("Testing queries with improved GraphRAG...")

        for i, query in enumerate(test_queries, 1):
            logger.info(f"\n--- Query {i}: {query} ---")

            try:
                # Route the query
                top_graphs, scores = router.route(query, k=2)
                logger.info(f"Routed to graphs: {list(zip(top_graphs, scores.tolist()))}")

                # Perform full query
                result = await router.query(query, k=2)

                if result and result.get("response"):
                    logger.info(f"Response: {result['response'][:200]}...")
                    logger.info(f"Metadata: {result['metadata']}")
                    logger.info(f"Sources: {len(result.get('source_nodes', []))} nodes")
                else:
                    logger.warning("No response generated")

            except Exception as e:
                logger.error(f"Error processing query: {e}")

        # Test direct retrieval
        logger.info("\n--- Testing direct retrieval ---")
        try:
            nodes = await router.retrieve_and_rerank("fire starting techniques", ["survivalist"])
            logger.info(f"Retrieved {len(nodes)} nodes directly")

            for i, node in enumerate(nodes[:3]):
                content = (
                    node.get_content()[:100] if hasattr(node, "get_content") else str(node)[:100]
                )
                score = getattr(node, "score", "N/A")
                logger.info(f"Node {i+1}: Score={score}, Content={content}...")

        except Exception as e:
            logger.error(f"Error in direct retrieval: {e}")

        # Clean up
        await router.close()
        logger.info("GraphRouter test completed successfully")
        return True

    except Exception as e:
        logger.error(f"Test failed with error: {e}")
        return False


if __name__ == "__main__":
    success = asyncio.run(test_improved_graphrag())
    sys.exit(0 if success else 1)
