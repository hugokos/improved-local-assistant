"""
Tests for the LLM Orchestration system.

This module tests the core orchestration functionality including semaphore
behavior, turn-by-turn coordination, and resource management.
"""

import asyncio
from unittest.mock import AsyncMock
from unittest.mock import MagicMock
from unittest.mock import patch

import pytest
from services.connection_pool_manager import ConnectionPoolManager
from services.llm_orchestrator import LLMOrchestrator
from services.working_set_cache import WorkingSetCache


@pytest.fixture
def mock_config():
    """Mock configuration for testing."""
    return {
        "orchestration": {
            "llm_semaphore_timeout": 5.0,
            "extraction_skip_on_pressure": True,
            "keep_alive_hermes": "30m",
            "keep_alive_tinyllama": 0,
            "json_mode_for_extraction": True,
            "max_extraction_time": 4.0,
        },
        "models": {"conversation": {"name": "hermes3:3b"}, "knowledge": {"name": "tinyllama"}},
        "connection_pool": {
            "max_connections": 5,
            "max_keepalive_connections": 2,
            "keepalive_expiry": 30.0,
            "timeout": {"connect": 5.0, "read": 30.0, "write": 30.0, "pool": 5.0},
        },
        "system": {"memory_threshold_percent": 80, "cpu_threshold_percent": 80},
    }


@pytest.fixture
def mock_system_monitor():
    """Mock system monitor for testing."""
    monitor = MagicMock()
    monitor.memory_pressure = 0.5
    monitor.cpu_usage = 0.5
    return monitor


@pytest.fixture
async def orchestrator(mock_config, mock_system_monitor):
    """Create orchestrator instance for testing."""
    orchestrator = LLMOrchestrator(mock_config, mock_system_monitor)

    # Mock the connection pool
    orchestrator.connection_pool = AsyncMock()
    orchestrator.connection_pool.initialize = AsyncMock()
    orchestrator.connection_pool.stream_chat = AsyncMock()
    orchestrator.connection_pool.chat_request = AsyncMock()

    await orchestrator.initialize()
    yield orchestrator
    await orchestrator.shutdown()


class TestLLMOrchestrator:
    """Test cases for LLM Orchestrator."""

    @pytest.mark.asyncio
    async def test_semaphore_prevents_concurrent_execution(self, orchestrator):
        """Test that the semaphore prevents concurrent LLM execution."""

        # Mock streaming response
        async def mock_stream():
            yield "Hello"
            await asyncio.sleep(0.1)  # Simulate processing time
            yield " world"

        orchestrator.connection_pool.stream_chat.return_value = mock_stream()

        # Start two concurrent turns
        task1 = asyncio.create_task(orchestrator.process_turn("session1", "Hello", []))
        task2 = asyncio.create_task(orchestrator.process_turn("session2", "Hi", []))

        # Collect results
        results1 = []
        results2 = []

        async for token in task1:
            results1.append(token)

        async for token in task2:
            results2.append(token)

        # Both should complete successfully
        assert results1 == ["Hello", " world"]
        assert results2 == ["Hello", " world"]

        # Verify semaphore was used (calls should be sequential)
        assert orchestrator.connection_pool.stream_chat.call_count == 2

    @pytest.mark.asyncio
    async def test_extraction_skipped_on_pressure(self, orchestrator, mock_system_monitor):
        """Test that extraction is skipped under resource pressure."""
        # Set high resource pressure
        mock_system_monitor.memory_pressure = 0.9
        mock_system_monitor.cpu_usage = 0.9

        # Mock streaming response
        async def mock_stream():
            yield "Response"

        orchestrator.connection_pool.stream_chat.return_value = mock_stream()

        # Process turn
        results = []
        async for token in orchestrator.process_turn("session1", "Test", []):
            results.append(token)

        # Should complete conversation but skip extraction
        assert results == ["Response"]
        assert orchestrator.metrics["extractions_skipped"] > 0

    @pytest.mark.asyncio
    async def test_extraction_cancellation_on_new_turn(self, orchestrator):
        """Test that active extraction is cancelled when new turn arrives."""

        # Mock slow extraction
        async def slow_extraction(*args, **kwargs):
            await asyncio.sleep(1.0)  # Simulate slow extraction
            return {"message": {"content": "[]"}}

        orchestrator.connection_pool.chat_request.side_effect = slow_extraction

        # Mock streaming response
        async def mock_stream():
            yield "Response"

        orchestrator.connection_pool.stream_chat.return_value = mock_stream()

        # Start first turn
        task1 = asyncio.create_task(orchestrator.process_turn("session1", "First", []))

        # Consume first response
        results1 = []
        async for token in task1:
            results1.append(token)

        # Start second turn immediately (should cancel first extraction)
        results2 = []
        async for token in orchestrator.process_turn("session1", "Second", []):
            results2.append(token)

        # Both conversations should complete
        assert results1 == ["Response"]
        assert results2 == ["Response"]

    @pytest.mark.asyncio
    async def test_model_prewarming(self, orchestrator):
        """Test that Hermes is pre-warmed during initialization."""
        # Verify pre-warming was called during initialization
        orchestrator.connection_pool.chat_request.assert_called()

        # Check that pre-warming used correct parameters
        call_args = orchestrator.connection_pool.chat_request.call_args
        assert call_args[1]["model"] == "hermes3:3b"
        assert call_args[1]["keep_alive"] == "30m"

    @pytest.mark.asyncio
    async def test_tinyllama_unloading(self, orchestrator):
        """Test that TinyLlama is unloaded after extraction."""
        # Mock extraction response
        orchestrator.connection_pool.chat_request.return_value = {"message": {"content": "[]"}}

        # Mock streaming response
        async def mock_stream():
            yield "Response"

        orchestrator.connection_pool.stream_chat.return_value = mock_stream()

        # Process turn with extraction
        results = []
        async for token in orchestrator.process_turn("session1", "Test", []):
            results.append(token)

        # Wait for background extraction to complete
        await asyncio.sleep(0.1)

        # Verify TinyLlama was unloaded (keep_alive=0)
        unload_calls = [
            call
            for call in orchestrator.connection_pool.chat_request.call_args_list
            if call[1].get("keep_alive") == 0
        ]
        assert len(unload_calls) >= 1


class TestConnectionPoolManager:
    """Test cases for Connection Pool Manager."""

    @pytest.mark.asyncio
    async def test_connection_pool_initialization(self, mock_config):
        """Test connection pool initialization with proper limits."""
        pool_manager = ConnectionPoolManager(mock_config)

        with patch("httpx.AsyncClient") as mock_client:
            await pool_manager.initialize()

            # Verify client was created with correct parameters
            mock_client.assert_called_once()
            call_kwargs = mock_client.call_args[1]

            assert call_kwargs["base_url"] == "http://localhost:11434"
            assert call_kwargs["http2"] is True

            # Check limits
            limits = call_kwargs["limits"]
            assert limits.max_connections == 5
            assert limits.max_keepalive_connections == 2

    @pytest.mark.asyncio
    async def test_keep_alive_parameter_handling(self, mock_config):
        """Test that keep_alive parameters are handled correctly."""
        pool_manager = ConnectionPoolManager(mock_config)
        pool_manager._client = AsyncMock()

        # Mock successful response
        mock_response = AsyncMock()
        mock_response.json.return_value = {"message": {"content": "test"}}
        pool_manager._client.post.return_value = mock_response

        # Test with explicit keep_alive
        await pool_manager.chat_request(
            model="hermes3:3b", messages=[{"role": "user", "content": "test"}], keep_alive="30m"
        )

        # Verify keep_alive was included in payload
        call_args = pool_manager._client.post.call_args
        payload = call_args[1]["json"]
        assert payload["keep_alive"] == "30m"

    @pytest.mark.asyncio
    async def test_model_residency_verification(self, mock_config):
        """Test model residency verification via /api/ps."""
        pool_manager = ConnectionPoolManager(mock_config)
        pool_manager._client = AsyncMock()

        # Mock /api/ps response
        mock_response = AsyncMock()
        mock_response.json.return_value = {
            "models": [
                {"name": "hermes3:3b", "size": 1000000},
                {"name": "tinyllama", "size": 500000},
            ]
        }
        pool_manager._client.get.return_value = mock_response

        # Test residency check
        is_resident = await pool_manager.verify_model_residency("hermes3:3b")
        assert is_resident is True

        is_resident = await pool_manager.verify_model_residency("nonexistent")
        assert is_resident is False


class TestWorkingSetCache:
    """Test cases for Working Set Cache."""

    @pytest.mark.asyncio
    async def test_session_cache_creation(self, mock_config):
        """Test that session caches are created on demand."""
        cache = WorkingSetCache(mock_config)
        await cache.initialize()

        # Get working set for new session
        working_set = await cache.get_working_set("session1")
        assert isinstance(working_set, set)
        assert len(working_set) == 0

        # Verify session was created
        assert "session1" in cache._session_caches

    @pytest.mark.asyncio
    async def test_lru_eviction(self, mock_config):
        """Test LRU eviction when session limits are exceeded."""
        # Set small limit for testing
        config = mock_config.copy()
        config["working_set_cache"] = {"nodes_per_session": 3}

        cache = WorkingSetCache(config)
        await cache.initialize()

        # Add nodes beyond limit
        await cache.update_working_set("session1", ["node1", "node2", "node3", "node4", "node5"])

        # Should only keep the most recent nodes
        working_set = await cache.get_working_set("session1")
        assert len(working_set) <= 3

        # Most recent nodes should be kept
        assert "node5" in working_set
        assert "node4" in working_set
        assert "node3" in working_set

    @pytest.mark.asyncio
    async def test_cache_persistence(self, mock_config, tmp_path):
        """Test cache persistence to disk."""
        # Use temporary directory
        config = mock_config.copy()
        config["working_set_cache"] = {"persist_dir": str(tmp_path)}

        cache = WorkingSetCache(config)
        await cache.initialize()

        # Add some data
        await cache.update_working_set("session1", ["node1", "node2"])

        # Persist cache
        await cache.persist_cache()

        # Verify file was created
        cache_file = tmp_path / "working_set_cache.json"
        assert cache_file.exists()

        # Create new cache instance and load
        cache2 = WorkingSetCache(config)
        await cache2.initialize()

        # Verify data was loaded
        working_set = await cache2.get_working_set("session1")
        assert "node1" in working_set
        assert "node2" in working_set


if __name__ == "__main__":
    pytest.main([__file__])
