"""
Pytest configuration and fixtures for the improved local assistant tests.
"""

import asyncio
import sys
from pathlib import Path

import pytest

# Add project root to Python path for imports
project_root = Path(__file__).parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from tests.mock_ollama import MockAsyncClient


@pytest.fixture
def mock_ollama_client():
    """Provide a mock Ollama client for testing."""
    return MockAsyncClient()


@pytest.fixture
async def async_mock_ollama_client():
    """Provide an async mock Ollama client for testing."""
    client = MockAsyncClient()
    yield client
    # Cleanup if needed


@pytest.fixture
def mock_config():
    """Provide a mock configuration for testing."""
    return {
        "ollama": {
            "host": "http://localhost:11434",
            "timeout": 30,
            "max_parallel": 2,
            "max_loaded_models": 2,
        },
        "models": {
            "conversation": {
                "name": "hermes3:3b",
                "context_window": 8000,
                "temperature": 0.7,
                "max_tokens": 2048,
            },
            "knowledge": {
                "name": "tinyllama",
                "context_window": 2048,
                "temperature": 0.2,
                "max_tokens": 1024,
            },
        },
        "knowledge_graphs": {
            "prebuilt_directory": "./data/prebuilt_graphs",
            "dynamic_storage": "./data/dynamic_graph",
            "max_triplets_per_chunk": 4,
            "enable_visualization": True,
        },
        "conversation": {
            "max_history_length": 50,
            "summarize_threshold": 20,
            "context_window_tokens": 8000,
        },
        "system": {
            "max_memory_gb": 12,
            "cpu_cores": 4,
            "memory_threshold_percent": 80,
            "cpu_threshold_percent": 80,
        },
        "api": {"host": "0.0.0.0", "port": 8000, "cors_origins": ["*"]},
    }


@pytest.fixture(scope="session")
def event_loop():
    """Create an instance of the default event loop for the test session."""
    loop = asyncio.get_event_loop_policy().new_event_loop()
    yield loop
    loop.close()


@pytest.fixture
def temp_data_dir(tmp_path):
    """Create temporary directories for test data."""
    data_dir = tmp_path / "data"
    data_dir.mkdir()

    # Create subdirectories
    (data_dir / "prebuilt_graphs").mkdir()
    (data_dir / "dynamic_graph").mkdir()
    (data_dir / "sessions").mkdir()
    (data_dir / "test_docs").mkdir()

    return data_dir


@pytest.fixture
def sample_documents(temp_data_dir):
    """Create sample documents for testing."""
    docs_dir = temp_data_dir / "test_docs"

    # Create sample text files
    (docs_dir / "sample1.txt").write_text("This is a sample document about Python programming.")
    (docs_dir / "sample2.txt").write_text(
        "Knowledge graphs are useful for representing relationships."
    )
    (docs_dir / "sample3.txt").write_text("Machine learning models can process natural language.")

    return docs_dir


# Pytest configuration
def pytest_configure(config):
    """Configure pytest with custom markers."""
    config.addinivalue_line(
        "markers", "slow: marks tests as slow (deselect with '-m \"not slow\"')"
    )
    config.addinivalue_line("markers", "integration: marks tests as integration tests")
    config.addinivalue_line("markers", "unit: marks tests as unit tests")


def pytest_collection_modifyitems(config, items):
    """Modify test collection to add markers automatically."""
    for item in items:
        # Mark tests in integration directories
        if "integration" in str(item.fspath):
            item.add_marker(pytest.mark.integration)
        # Mark slow tests
        if "slow" in item.name or "integration" in str(item.fspath):
            item.add_marker(pytest.mark.slow)
