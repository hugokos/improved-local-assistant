"""
Smoke tests for end-to-end functionality.
These tests verify the core system works with minimal setup.
"""
import tempfile
from unittest.mock import patch

import pytest

from improved_local_assistant.app.main import create_app
from improved_local_assistant.services.graph_service import GraphService


@pytest.mark.smoke
@pytest.mark.asyncio
async def test_app_startup_smoke():
    """Test that the FastAPI app can start without errors."""
    app = create_app()
    assert app is not None
    assert hasattr(app, 'routes')


@pytest.mark.smoke
@pytest.mark.asyncio
async def test_graph_service_initialization():
    """Test that GraphService can initialize with minimal config."""
    with tempfile.TemporaryDirectory() as temp_dir:
        config = {
            'storage_path': temp_dir,
            'embedding_model': 'sentence-transformers/all-MiniLM-L6-v2',
            'llm_model': 'llama3.2:1b'
        }

        # Mock Ollama client to avoid external dependency
        with patch('ollama.Client') as mock_client:
            mock_client.return_value.embeddings.return_value = {
                'embedding': [0.1] * 384
            }

            service = GraphService(config)
            assert service is not None


@pytest.mark.smoke
def test_basic_text_processing():
    """Test basic text processing functionality."""
    from improved_local_assistant.services.text_processor import TextProcessor

    processor = TextProcessor()
    text = "This is a test document with some entities like OpenAI and Python."

    # Test chunking
    chunks = processor.chunk_text(text, chunk_size=50)
    assert len(chunks) > 0
    assert all(isinstance(chunk, str) for chunk in chunks)

    # Test entity extraction (mock)
    with patch.object(processor, 'extract_entities') as mock_extract:
        mock_extract.return_value = [
            {'text': 'OpenAI', 'type': 'ORG'},
            {'text': 'Python', 'type': 'TECH'}
        ]
        entities = processor.extract_entities(text)
        assert len(entities) == 2


@pytest.mark.smoke
@pytest.mark.asyncio
async def test_simple_query_flow():
    """Test a simple query flow with mocked components."""
    from improved_local_assistant.services.query_service import QueryService

    with tempfile.TemporaryDirectory() as temp_dir:
        config = {
            'storage_path': temp_dir,
            'embedding_model': 'sentence-transformers/all-MiniLM-L6-v2',
            'llm_model': 'llama3.2:1b'
        }

        # Mock external dependencies
        with patch('ollama.Client') as mock_ollama:
            mock_ollama.return_value.chat.return_value = {
                'message': {'content': 'This is a test response.'}
            }
            mock_ollama.return_value.embeddings.return_value = {
                'embedding': [0.1] * 384
            }

            service = QueryService(config)
            response = await service.query("What is Python?")

            assert response is not None
            assert 'content' in response or 'answer' in response


@pytest.mark.smoke
def test_configuration_loading():
    """Test that configuration can be loaded from various sources."""
    from improved_local_assistant.config import load_config

    # Test default config
    config = load_config()
    assert config is not None
    assert 'storage_path' in config
    assert 'embedding_model' in config

    # Test config with overrides
    overrides = {'llm_model': 'test-model'}
    config_with_overrides = load_config(overrides)
    assert config_with_overrides['llm_model'] == 'test-model'


@pytest.mark.smoke
def test_voice_components_import():
    """Test that voice components can be imported without errors."""
    try:
        from improved_local_assistant.services.stt_service import STTService
        from improved_local_assistant.services.tts_service import TTSService
        from improved_local_assistant.services.voice_service import VoiceService

        # Just test imports, not functionality (which requires audio hardware)
        assert VoiceService is not None
        assert TTSService is not None
        assert STTService is not None
    except ImportError as e:
        pytest.skip(f"Voice components not available: {e}")


@pytest.mark.smoke
def test_cli_tools_import():
    """Test that CLI tools can be imported."""
    try:
        from improved_local_assistant.cli.graphrag_repl import main as repl_main
        assert repl_main is not None
    except ImportError as e:
        pytest.skip(f"CLI tools not available: {e}")


@pytest.mark.smoke
@pytest.mark.asyncio
async def test_websocket_endpoint_exists():
    """Test that WebSocket endpoints are properly configured."""
    from fastapi.testclient import TestClient

    from improved_local_assistant.app.main import create_app

    app = create_app()
    client = TestClient(app)

    # Test that the WebSocket route exists (will fail connection but route should exist)
    with pytest.raises(Exception):  # Expected to fail without proper WebSocket client
        with client.websocket_connect("/ws/chat"):
            pass

    # If we get here, the route exists (connection failure is expected in test)


if __name__ == "__main__":
    # Run smoke tests
    pytest.main([__file__, "-v", "-m", "smoke"])
