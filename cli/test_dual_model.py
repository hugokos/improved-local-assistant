#!/usr/bin/env python
"""
Test script for the dual-model architecture with async processing.

This script tests concurrent processing, fire-and-forget background tasks,
resource isolation, and non-blocking operations.
"""

import argparse
import asyncio
import logging
import os
import sys

# Fix for Windows asyncio event loop issue
if sys.platform == "win32":
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
import time
from pathlib import Path

from dotenv import load_dotenv

# Add parent directory to path to import from services
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

# Check if we should use mock
USE_MOCK = os.getenv("USE_MOCK", "true").lower() == "true"

if USE_MOCK:
    # Import mock client
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "tests"))
    # Monkey patch the AsyncClient in model_mgr
from improved_local_assistant import services.model_mgr
    from mock_ollama import MockAsyncClient

    services.model_mgr.AsyncClient = MockAsyncClient
    print("Using mock Ollama client for testing")

from improved_local_assistant.services.model_mgr import ModelConfig  # noqa: E402
from improved_local_assistant.services.model_mgr import ModelManager  # noqa: E402

# Load environment variables from .env file
load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


async def test_concurrent_processing():
    """Test concurrent processing of conversation and background models."""
    logger.info("Testing concurrent processing...")

    # Get configuration from environment variables
    ollama_host = os.getenv("OLLAMA_HOST", "http://localhost:11434")
    max_parallel = int(os.getenv("OLLAMA_NUM_PARALLEL", "2"))
    max_loaded = int(os.getenv("OLLAMA_MAX_LOADED_MODELS", "2"))

    model_manager = ModelManager(host=ollama_host)
    config = ModelConfig(name="test", type="test", max_parallel=max_parallel, max_loaded=max_loaded)

    # Initialize models
    success = await model_manager.initialize_models(config)
    if not success:
        logger.error("❌ Failed to initialize models!")
        return False

    # Disable load monitoring for this test to avoid task cancellation
    if hasattr(model_manager, "_load_check_task") and model_manager._load_check_task:
        await model_manager.stop_load_monitoring()

    # Test concurrent operation
    test_message = (
        "Explain how knowledge graphs can be used with LLMs for better context retrieval."
    )

    logger.info("Running concurrent queries...")
    conversation_stream, bg_task = await model_manager.run_concurrent_queries(
        test_message, priority_mode=False
    )

    print("\n--- Conversation Response ---")
    async for token in conversation_stream:
        print(token, end="", flush=True)
    print("\n---------------------------\n")

    # Wait for background task to complete
    try:
        bg_result = await bg_task

        print("\n--- Background Knowledge Extraction ---")
        print(bg_result["content"])
        print(f"Elapsed time: {bg_result['elapsed_time']:.2f} seconds")
        print("-------------------------------------\n")
    except asyncio.CancelledError:
        print("\n--- Background task was cancelled (expected in priority mode) ---")
        print("This demonstrates the fire-and-forget pattern where background tasks")
        print("can be cancelled if system resources are needed elsewhere.")
        print("-------------------------------------\n")

    # Get model status after concurrent operation
    status = await model_manager.get_model_status()
    logger.info(f"Conversation metrics: {status['metrics']['conversation']}")
    logger.info(f"Knowledge metrics: {status['metrics']['knowledge']}")

    logger.info("✅ Concurrent operation test completed successfully!")
    return True


async def test_fire_and_forget():
    """Test fire-and-forget background task pattern."""
    logger.info("Testing fire-and-forget pattern...")

    # Get configuration from environment variables
    ollama_host = os.getenv("OLLAMA_HOST", "http://localhost:11434")
    max_parallel = int(os.getenv("OLLAMA_NUM_PARALLEL", "2"))
    max_loaded = int(os.getenv("OLLAMA_MAX_LOADED_MODELS", "2"))

    model_manager = ModelManager(host=ollama_host)
    config = ModelConfig(name="test", type="test", max_parallel=max_parallel, max_loaded=max_loaded)

    # Initialize models
    success = await model_manager.initialize_models(config)
    if not success:
        logger.error("❌ Failed to initialize models!")
        return False

    # Test fire-and-forget pattern
    test_message = "The TinyLlama model is efficient for background processing tasks."

    logger.info("Running fire-and-forget background task...")
    conversation_stream, bg_task = await model_manager.run_concurrent_queries(test_message)

    print("\n--- Processing conversation without waiting for background task ---")
    async for token in conversation_stream:
        print(token, end="", flush=True)
    print("\n----------------------------------------------------------\n")

    print("Conversation completed. Background task is still running...")
    print(f"Background task done: {bg_task.done()}")

    # Wait a bit to see background task progress
    await asyncio.sleep(2)
    print(f"After 2 seconds - Background task done: {bg_task.done()}")

    # Now wait for background task to complete
    try:
        bg_result = await bg_task

        print("\n--- Background Knowledge Extraction (completed after conversation) ---")
        print(bg_result["content"])
        print(f"Elapsed time: {bg_result['elapsed_time']:.2f} seconds")
        print("----------------------------------------------------------------\n")
    except asyncio.CancelledError:
        print("\n--- Background task was cancelled (expected in priority mode) ---")
        print("This demonstrates the fire-and-forget pattern where background tasks")
        print("can be cancelled if system resources are needed elsewhere.")
        print("----------------------------------------------------------------\n")

    logger.info("✅ Fire-and-forget pattern test completed successfully!")
    return True


async def test_resource_isolation():
    """Test resource isolation between models."""
    logger.info("Testing resource isolation...")

    # Get configuration from environment variables
    ollama_host = os.getenv("OLLAMA_HOST", "http://localhost:11434")
    max_parallel = int(os.getenv("OLLAMA_NUM_PARALLEL", "2"))
    max_loaded = int(os.getenv("OLLAMA_MAX_LOADED_MODELS", "2"))

    model_manager = ModelManager(host=ollama_host)
    config = ModelConfig(name="test", type="test", max_parallel=max_parallel, max_loaded=max_loaded)

    # Initialize models
    success = await model_manager.initialize_models(config)
    if not success:
        logger.error("❌ Failed to initialize models!")
        return False

    # Test resource isolation with multiple concurrent tasks
    print("\n--- Testing resource isolation with multiple concurrent tasks ---")

    # Create multiple background tasks
    bg_tasks = []
    for i in range(3):
        text = f"Topic {i+1} is about testing resource isolation between multiple model instances."
        task = asyncio.create_task(model_manager.query_knowledge_model(text))
        bg_tasks.append(task)

    # While background tasks are running, run a conversation task
    messages = [{"role": "user", "content": "Write a short paragraph about resource isolation."}]

    print("Running conversation task while background tasks are processing...")
    response_text = ""
    async for token in model_manager.query_conversation_model(messages):
        response_text += token

    print("\n--- Conversation Response (should be responsive despite background load) ---")
    print(response_text)
    print("----------------------------------------------------------------------\n")

    # Wait for all background tasks to complete
    results = await asyncio.gather(*bg_tasks)

    print("\n--- Background Task Results ---")
    for i, result in enumerate(results):
        print(f"Task {i+1} completed in {result['elapsed_time']:.2f} seconds")
    print("-----------------------------\n")

    logger.info("✅ Resource isolation test completed successfully!")
    return True


async def test_non_blocking():
    """Test that operations don't block each other."""
    logger.info("Testing non-blocking operations...")

    # Get configuration from environment variables
    ollama_host = os.getenv("OLLAMA_HOST", "http://localhost:11434")
    max_parallel = int(os.getenv("OLLAMA_NUM_PARALLEL", "2"))
    max_loaded = int(os.getenv("OLLAMA_MAX_LOADED_MODELS", "2"))

    model_manager = ModelManager(host=ollama_host)
    config = ModelConfig(name="test", type="test", max_parallel=max_parallel, max_loaded=max_loaded)

    # Initialize models
    success = await model_manager.initialize_models(config)
    if not success:
        logger.error("❌ Failed to initialize models!")
        return False

    # Start a long-running background task
    print("\n--- Testing non-blocking operations ---")
    print("Starting a long-running task...")

    long_text = (
        "Write a detailed essay about the history of artificial intelligence, "
        "including major milestones, key researchers, and future directions."
    )
    messages = [{"role": "user", "content": long_text}]

    # Start the long-running task
    long_task_stream = model_manager.query_conversation_model(messages)

    # Start a short task immediately after
    print("Starting a short task while long task is running...")
    short_messages = [{"role": "user", "content": "What is 2+2?"}]
    short_task_start = time.time()

    short_response = ""
    async for token in model_manager.query_conversation_model(short_messages):
        short_response += token

    short_task_duration = time.time() - short_task_start

    print("\n--- Short Task Response ---")
    print(short_response)
    print(f"Short task completed in {short_task_duration:.2f} seconds")
    print("--------------------------\n")

    # Clean up the long-running task by consuming its output
    print("Now waiting for long task to complete...")
    long_response = ""
    async for token in long_task_stream:
        long_response += token

    print("\n--- Long Task Completed ---")
    print(f"Long response length: {len(long_response)} characters")
    print("-------------------------\n")

    logger.info("✅ Non-blocking operations test completed successfully!")
    return True


async def test_priority_management():
    """Test priority management during high load."""
    logger.info("Testing priority management...")

    # Get configuration from environment variables
    ollama_host = os.getenv("OLLAMA_HOST", "http://localhost:11434")
    max_parallel = int(os.getenv("OLLAMA_NUM_PARALLEL", "2"))
    max_loaded = int(os.getenv("OLLAMA_MAX_LOADED_MODELS", "2"))

    model_manager = ModelManager(host=ollama_host)
    config = ModelConfig(name="test", type="test", max_parallel=max_parallel, max_loaded=max_loaded)

    # Initialize models
    success = await model_manager.initialize_models(config)
    if not success:
        logger.error("❌ Failed to initialize models!")
        return False

    # Start load monitoring
    await model_manager.start_load_monitoring()

    print("\n--- Testing priority management during high load ---")
    print("Creating artificial high load with multiple background tasks...")

    # Create multiple background tasks to simulate high load
    bg_tasks = []
    for i in range(5):
        text = f"Generate a detailed analysis of topic {i+1} with multiple paragraphs and complex reasoning."
        task = asyncio.create_task(model_manager.query_knowledge_model(text))
        bg_tasks.append(task)

    # Wait a moment for load to build
    await asyncio.sleep(2)

    # Run a conversation task that should get priority
    print("Running conversation task that should get priority...")
    messages = [
        {
            "role": "user",
            "content": "What's the most important aspect of resource management in AI systems?",
        }
    ]

    start_time = time.time()
    response_text = ""
    async for token in model_manager.query_conversation_model(messages):
        response_text += token

    conversation_duration = time.time() - start_time

    print("\n--- Conversation Response (should be responsive despite high load) ---")
    print(response_text)
    print(f"Conversation completed in {conversation_duration:.2f} seconds")
    print("----------------------------------------------------------------------\n")

    # Wait for background tasks to complete or cancel them after timeout
    try:
        await asyncio.wait_for(asyncio.gather(*bg_tasks), timeout=10)
    except asyncio.TimeoutError:
        print("Some background tasks took too long and were cancelled")
        for task in bg_tasks:
            if not task.done():
                task.cancel()

    # Stop load monitoring
    await model_manager.stop_load_monitoring()

    logger.info("✅ Priority management test completed successfully!")
    return True


async def run_all_tests():
    """Run all dual-model architecture tests."""
    tests = [
        ("Concurrent Processing", test_concurrent_processing),
        ("Fire-and-Forget Pattern", test_fire_and_forget),
        ("Resource Isolation", test_resource_isolation),
        ("Non-Blocking Operations", test_non_blocking),
        ("Priority Management", test_priority_management),
    ]

    results = {}

    for name, test_func in tests:
        print(f"\n{'=' * 80}")
        print(f"Running Test: {name}")
        print(f"{'=' * 80}\n")

        try:
            start_time = time.time()
            success = await test_func()
            elapsed = time.time() - start_time

            if success:
                results[name] = f"✅ PASSED ({elapsed:.2f}s)"
            else:
                results[name] = f"❌ FAILED ({elapsed:.2f}s)"
        except Exception as e:
            results[name] = f"❌ ERROR: {str(e)}"

    # Print summary
    print(f"\n{'=' * 80}")
    print("Test Summary")
    print(f"{'=' * 80}")

    for name, result in results.items():
        print(f"{name}: {result}")

    # Return overall success
    return all(result.startswith("✅") for result in results.values())


def main():
    """Main entry point for the script."""
    parser = argparse.ArgumentParser(
        description="Test dual-model architecture with async processing"
    )
    parser.add_argument("--concurrent", action="store_true", help="Test concurrent processing")
    parser.add_argument("--fire-forget", action="store_true", help="Test fire-and-forget pattern")
    parser.add_argument("--isolation", action="store_true", help="Test resource isolation")
    parser.add_argument("--non-blocking", action="store_true", help="Test non-blocking operations")
    parser.add_argument("--priority", action="store_true", help="Test priority management")
    parser.add_argument("--all", action="store_true", help="Run all tests")

    args = parser.parse_args()

    # If no specific test is selected, run all tests
    if not any(
        [
            args.concurrent,
            args.fire_forget,
            args.isolation,
            args.non_blocking,
            args.priority,
            args.all,
        ]
    ):
        args.all = True

    async def run_selected_tests():
        if args.all:
            return await run_all_tests()

        success = True

        if args.concurrent:
            success = success and await test_concurrent_processing()

        if args.fire_forget:
            success = success and await test_fire_and_forget()

        if args.isolation:
            success = success and await test_resource_isolation()

        if args.non_blocking:
            success = success and await test_non_blocking()

        if args.priority:
            success = success and await test_priority_management()

        return success

    success = asyncio.run(run_selected_tests())
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
