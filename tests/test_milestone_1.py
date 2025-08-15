#!/usr/bin/env python
"""
Test script for Milestone 1: Model Integration and Basic Testing.

This script tests the dual-model architecture with async processing,
focusing on concurrent model operations, fire-and-forget background tasks,
resource isolation, and non-blocking operations.
"""

import asyncio
import logging
import os
import sys
import time

import pytest
import pytest_asyncio

# Fix for Windows asyncio event loop issue
if sys.platform == "win32":
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
from pathlib import Path

from dotenv import load_dotenv

# Add parent directory to path to import from services
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from services.model_mgr import ModelConfig
from services.model_mgr import ModelManager

# Check if Ollama is available, otherwise use mock
USE_MOCK = os.getenv("USE_MOCK", "true").lower() == "true"

if USE_MOCK:
    # Monkey patch the AsyncClient in model_mgr
    import services.model_mgr

    from tests.mock_ollama import MockAsyncClient

    services.model_mgr.AsyncClient = MockAsyncClient


# Load environment variables from .env file
load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


class TestMilestone1:
    """Test cases for Milestone 1: Model Integration and Basic Testing."""

    @pytest_asyncio.fixture(scope="class")
    async def model_manager(self):
        """Fixture to create and initialize a ModelManager instance."""
        # Get configuration from environment variables
        ollama_host = os.getenv("OLLAMA_HOST", "http://localhost:11434")
        max_parallel = int(os.getenv("OLLAMA_NUM_PARALLEL", "2"))
        max_loaded = int(os.getenv("OLLAMA_MAX_LOADED_MODELS", "2"))

        model_manager = ModelManager(host=ollama_host)
        config = ModelConfig(
            name="test", type="test", max_parallel=max_parallel, max_loaded=max_loaded
        )

        # Initialize models
        success = await model_manager.initialize_models(config)
        assert success, "Failed to initialize models"

        yield model_manager  # Use yield instead of return for fixtures

    @pytest.mark.asyncio
    async def test_dual_model_architecture(self, model_manager):
        """Test that both models can be initialized and used."""
        # Test conversation model
        messages = [{"role": "user", "content": "Hello, how are you?"}]
        response_text = ""
        async for token in model_manager.query_conversation_model(messages):
            response_text += token

        assert response_text, "Conversation model should return a response"

        # Test knowledge model
        test_text = "The Hermes 3:3B model was developed by Nous Research."
        result = await model_manager.query_knowledge_model(test_text)

        assert result["content"], "Knowledge model should return a response"
        assert "elapsed_time" in result, "Response should include elapsed time"

    @pytest.mark.asyncio
    async def test_concurrent_processing(self, model_manager):
        """Test concurrent processing of conversation and background models."""
        test_message = "Explain how knowledge graphs can be used with LLMs."

        # Run concurrent queries
        conversation_stream, bg_task = await model_manager.run_concurrent_queries(test_message)

        # Process conversation stream
        response_text = ""
        async for token in conversation_stream:
            response_text += token

        # Wait for background task to complete
        bg_result = await bg_task

        assert response_text, "Conversation model should return a response"
        assert bg_result["content"], "Background task should return a result"
        assert "elapsed_time" in bg_result, "Background task should include elapsed time"

    @pytest.mark.asyncio
    async def test_fire_and_forget_pattern(self, model_manager):
        """Test fire-and-forget background task pattern."""
        test_message = "The TinyLlama model is efficient for background processing tasks."

        # Start a background task without awaiting it
        conversation_stream, bg_task = await model_manager.run_concurrent_queries(test_message)

        # Process conversation stream without waiting for background task
        response_text = ""
        async for token in conversation_stream:
            response_text += token

        assert response_text, "Conversation model should return a response"

        # Background task should still be running or completed
        assert not bg_task.cancelled(), "Background task should not be cancelled"

        # Now wait for background task to complete
        bg_result = await bg_task
        assert bg_result["content"], "Background task should eventually complete"

    @pytest.mark.asyncio
    async def test_resource_isolation(self, model_manager):
        """Test resource isolation between models."""
        # Run multiple concurrent operations to test resource isolation
        tasks = []

        # Create 3 conversation tasks
        for i in range(3):
            messages = [{"role": "user", "content": f"Write a short paragraph about topic {i+1}"}]
            tasks.append(self._collect_stream(model_manager.query_conversation_model(messages)))

        # Create 2 knowledge tasks
        for i in range(2):
            text = (
                f"Topic {i+1} is about testing resource isolation between multiple model instances."
            )
            tasks.append(model_manager.query_knowledge_model(text))

        # Run all tasks concurrently
        results = await asyncio.gather(*tasks, return_exceptions=True)

        # Check that all tasks completed without exceptions
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                pytest.fail(f"Task {i} failed with exception: {result}")
            else:
                assert result, f"Task {i} should return a result"

    @pytest.mark.asyncio
    async def test_non_blocking_operations(self, model_manager):
        """Test that operations don't block each other."""
        # Start a long-running background task
        long_text = (
            "Write a detailed essay about the history of artificial intelligence, "
            "including major milestones, key researchers, and future directions."
        )
        messages = [{"role": "user", "content": long_text}]

        # Start the long-running task
        long_task_stream = model_manager.query_conversation_model(messages)

        # Start a short task immediately after
        short_messages = [{"role": "user", "content": "What is 2+2?"}]
        short_task_start = time.time()
        short_response = ""
        async for token in model_manager.query_conversation_model(short_messages):
            short_response += token
        short_task_duration = time.time() - short_task_start

        # The short task should complete quickly even though the long task is running
        assert short_response, "Short task should return a response"
        assert short_task_duration < 10, "Short task should complete quickly"

        # Clean up the long-running task by consuming its output
        long_response = ""
        async for token in long_task_stream:
            long_response += token

        assert long_response, "Long task should eventually complete"

    async def _collect_stream(self, stream):
        """Helper method to collect all tokens from a stream."""
        result = ""
        async for token in stream:
            result += token
        return result


if __name__ == "__main__":
    """Run the tests directly."""
    import pytest

    pytest.main(["-xvs", __file__])
