#!/usr/bin/env python3
"""
Test script for the GraphRAG REPL to verify functionality.

This script tests the GraphRAG REPL components without requiring user interaction.
"""

import asyncio
import logging
import os
import sys

# Add current directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Fix for Windows asyncio event loop issue
if sys.platform == "win32":
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

from cli.graphrag_repl import GraphRAGREPL  # noqa: E402


async def test_initialization():
    """Test component initialization."""
    logger.info("Testing GraphRAG REPL initialization...")

    # Test with KG enabled
    repl_with_kg = GraphRAGREPL(use_kg=True, max_triple_per_chunk=3)

    try:
        # Test embedding model initialization
        embedding_success = await repl_with_kg.initialize_embedding_model()
        logger.info(
            f"Embedding model initialization: {'SUCCESS' if embedding_success else 'FAILED'}"
        )

        # Test model manager initialization
        model_success = await repl_with_kg.initialize_model_manager()
        logger.info(f"Model manager initialization: {'SUCCESS' if model_success else 'FAILED'}")

        if model_success:
            # Test KG manager initialization
            kg_success = await repl_with_kg.initialize_kg_manager()
            logger.info(f"KG manager initialization: {'SUCCESS' if kg_success else 'FAILED'}")

            if kg_success:
                # Test conversation manager initialization
                conv_success = await repl_with_kg.initialize_conversation_manager()
                logger.info(
                    f"Conversation manager initialization: {'SUCCESS' if conv_success else 'FAILED'}"
                )

                # Test resource manager initialization
                resource_success = await repl_with_kg.initialize_resource_manager()
                logger.info(
                    f"Resource manager initialization: {'SUCCESS' if resource_success else 'FAILED'}"
                )

                if conv_success and resource_success:
                    logger.info("✅ All components initialized successfully with KG enabled")

                    # Test a simple query
                    await test_simple_query(repl_with_kg)

                    # Cleanup
                    await repl_with_kg.handle_keyboard_interrupt()
                    return True

    except Exception as e:
        logger.error(f"Error during initialization test: {str(e)}")
        return False

    return False


async def test_simple_query(repl):
    """Test a simple query processing."""
    try:
        logger.info("Testing simple query processing...")

        # Simulate a simple query
        test_query = "Hello, can you tell me about artificial intelligence?"

        logger.info(f"Processing query: {test_query}")

        # Process the query (this will print to stdout)
        await repl.process_user_query(test_query)

        logger.info("✅ Simple query processed successfully")

    except Exception as e:
        logger.error(f"Error during query test: {str(e)}")


async def test_no_kg_mode():
    """Test GraphRAG REPL without knowledge graph."""
    logger.info("Testing GraphRAG REPL without knowledge graph...")

    repl_no_kg = GraphRAGREPL(use_kg=False, max_triple_per_chunk=3)

    try:
        # Initialize components (should skip KG)
        success = await repl_no_kg.initialize_all_components()

        if success:
            logger.info("✅ No-KG mode initialization successful")

            # Test a simple query without KG
            test_query = "What is machine learning?"
            logger.info(f"Processing query without KG: {test_query}")

            await repl_no_kg.process_user_query(test_query)

            logger.info("✅ No-KG query processed successfully")

            # Cleanup
            await repl_no_kg.handle_keyboard_interrupt()
            return True
        else:
            logger.error("❌ No-KG mode initialization failed")
            return False

    except Exception as e:
        logger.error(f"Error during no-KG test: {str(e)}")
        return False


async def main():
    """Main test function."""
    logger.info("Starting GraphRAG REPL tests...")

    # Test 1: Full initialization with KG
    init_success = await test_initialization()

    # Test 2: No-KG mode
    no_kg_success = await test_no_kg_mode()

    # Summary
    logger.info("\n" + "=" * 50)
    logger.info("TEST SUMMARY")
    logger.info("=" * 50)
    logger.info(f"Full initialization test: {'PASSED' if init_success else 'FAILED'}")
    logger.info(f"No-KG mode test: {'PASSED' if no_kg_success else 'FAILED'}")

    if init_success and no_kg_success:
        logger.info("✅ All tests passed! GraphRAG REPL is ready to use.")
        logger.info("\nTo run the interactive REPL:")
        logger.info("  python run_graphrag_repl.py")
        logger.info("  python run_graphrag_repl.py --no-kg")
        logger.info("  python run_graphrag_repl.py --max-triple-per-chunk 6")
    else:
        logger.info("❌ Some tests failed. Check the logs above for details.")
        logger.info("\nCommon issues:")
        logger.info("- Ensure Ollama is running: ollama serve")
        logger.info(
            "- Pull required models: ollama pull hermes3:3b && ollama pull tinyllama:latest"
        )
        logger.info("- Check requirements.txt dependencies are installed")


if __name__ == "__main__":
    asyncio.run(main())
