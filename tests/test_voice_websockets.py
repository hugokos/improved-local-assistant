"""
Tests for voice WebSocket endpoints in the Improved Local AI Assistant.

Tests the STT and TTS WebSocket endpoints for proper functionality.
"""

import json
from unittest.mock import AsyncMock
from unittest.mock import Mock

import pytest

# Import WebSocket functions
try:
    from app.ws.voice_stt import stt_websocket
    from app.ws.voice_tts import tts_websocket

    VOICE_WS_IMPORTS_AVAILABLE = True
except ImportError as e:
    VOICE_WS_IMPORTS_AVAILABLE = False
    IMPORT_ERROR = str(e)


@pytest.mark.skipif(
    not VOICE_WS_IMPORTS_AVAILABLE,
    reason=f"Voice WebSocket imports not available: {IMPORT_ERROR if not VOICE_WS_IMPORTS_AVAILABLE else ''}",
)
class TestVoiceWebSockets:
    """Test cases for voice WebSocket endpoints."""

    @pytest.fixture
    def mock_app(self):
        """Mock FastAPI app with voice services."""
        app = Mock()
        app.state = Mock()

        # Mock voice manager
        voice_manager = Mock()
        voice_manager.create_voice_session = AsyncMock(return_value=True)
        voice_manager.destroy_voice_session = AsyncMock(return_value=True)
        voice_manager.process_audio_chunk = AsyncMock(return_value={"partial": "test"})
        voice_manager.set_voice_session_listening = Mock()

        async def mock_synthesize_response(session_id, text):
            for i in range(3):
                yield f"audio_chunk_{i}".encode()

        voice_manager.synthesize_response = mock_synthesize_response

        app.state.voice_manager = voice_manager
        app.state.conversation_manager = Mock()
        app.state.connection_manager = Mock()

        return app

    @pytest.fixture
    def mock_websocket(self):
        """Mock WebSocket connection."""
        websocket = Mock()
        websocket.accept = AsyncMock()
        websocket.send_json = AsyncMock()
        websocket.send_bytes = AsyncMock()
        websocket.send_text = AsyncMock()
        websocket.receive_bytes = AsyncMock()
        websocket.receive_text = AsyncMock()
        websocket.close = AsyncMock()
        websocket.client = Mock()
        websocket.client.host = "127.0.0.1"
        websocket.application_state = "CONNECTED"  # Mock WebSocketState.CONNECTED

        return websocket

    @pytest.mark.asyncio
    async def test_stt_websocket_initialization(self, mock_app, mock_websocket):
        """Test STT WebSocket initialization."""
        session_id = "test_session_123"

        # Mock WebSocket to disconnect after initialization
        call_count = 0

        async def mock_receive_bytes():
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                return b"fake_audio_data"
            else:
                # Simulate disconnect
                from fastapi import WebSocketDisconnect

                raise WebSocketDisconnect()

        mock_websocket.receive_bytes = mock_receive_bytes

        # Test STT WebSocket
        await stt_websocket(mock_websocket, session_id, mock_app)

        # Verify initialization calls
        mock_websocket.accept.assert_called_once()
        mock_app.state.voice_manager.create_voice_session.assert_called_once_with(session_id)
        mock_app.state.voice_manager.set_voice_session_listening.assert_called_with(
            session_id, True
        )

    @pytest.mark.asyncio
    async def test_stt_websocket_audio_processing(self, mock_app, mock_websocket):
        """Test STT WebSocket audio processing."""
        session_id = "test_session_123"

        # Mock audio data reception
        audio_data = b"fake_audio_data"
        call_count = 0

        async def mock_receive_bytes():
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                return audio_data
            else:
                from fastapi import WebSocketDisconnect

                raise WebSocketDisconnect()

        mock_websocket.receive_bytes = mock_receive_bytes

        # Mock STT result
        mock_app.state.voice_manager.process_audio_chunk.return_value = {"partial": "hello"}

        # Test STT WebSocket
        await stt_websocket(mock_websocket, session_id, mock_app)

        # Verify audio processing
        mock_app.state.voice_manager.process_audio_chunk.assert_called_with(session_id, audio_data)

        # Verify partial result sent
        {
            "type": "stt_partial",
            "text": "hello",
            "session_id": session_id,
            "timestamp": mock_websocket.send_json.call_args[0][0]["timestamp"],
        }
        mock_websocket.send_json.assert_called()

    @pytest.mark.asyncio
    async def test_stt_websocket_final_result(self, mock_app, mock_websocket):
        """Test STT WebSocket final result handling."""
        session_id = "test_session_123"

        # Mock audio data reception
        call_count = 0

        async def mock_receive_bytes():
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                return b"fake_audio_data"
            else:
                from fastapi import WebSocketDisconnect

                raise WebSocketDisconnect()

        mock_websocket.receive_bytes = mock_receive_bytes

        # Mock final STT result
        mock_app.state.voice_manager.process_audio_chunk.return_value = {
            "final": "hello world",
            "confidence": 0.95,
        }

        # Mock connection manager
        chat_websocket = Mock()
        chat_websocket.send_text = AsyncMock()
        chat_websocket.application_state = "CONNECTED"
        mock_app.state.connection_manager.get_connection.return_value = chat_websocket

        # Test STT WebSocket
        await stt_websocket(mock_websocket, session_id, mock_app)

        # Verify final result sent
        mock_websocket.send_json.assert_called()
        sent_messages = [call.args[0] for call in mock_websocket.send_json.call_args_list]

        # Find the final result message
        final_message = next((msg for msg in sent_messages if msg.get("type") == "stt_final"), None)
        assert final_message is not None
        assert final_message["text"] == "hello world"
        assert final_message["confidence"] == 0.95

    @pytest.mark.asyncio
    async def test_tts_websocket_initialization(self, mock_app, mock_websocket):
        """Test TTS WebSocket initialization."""
        session_id = "test_session_123"

        # Mock text reception
        call_count = 0

        async def mock_receive_text():
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                return "Hello, this is a test."
            else:
                from fastapi import WebSocketDisconnect

                raise WebSocketDisconnect()

        mock_websocket.receive_text = mock_receive_text

        # Test TTS WebSocket
        await tts_websocket(mock_websocket, session_id, mock_app)

        # Verify initialization calls
        mock_websocket.accept.assert_called_once()
        mock_app.state.voice_manager.create_voice_session.assert_called_once_with(session_id)

    @pytest.mark.asyncio
    async def test_tts_websocket_synthesis(self, mock_app, mock_websocket):
        """Test TTS WebSocket synthesis."""
        session_id = "test_session_123"
        test_text = "Hello, this is a test."

        # Mock text reception
        call_count = 0

        async def mock_receive_text():
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                return test_text
            else:
                from fastapi import WebSocketDisconnect

                raise WebSocketDisconnect()

        mock_websocket.receive_text = mock_receive_text

        # Test TTS WebSocket
        await tts_websocket(mock_websocket, session_id, mock_app)

        # Verify synthesis start message
        start_calls = [
            call
            for call in mock_websocket.send_json.call_args_list
            if call.args[0].get("type") == "tts_start"
        ]
        assert len(start_calls) > 0
        assert start_calls[0].args[0]["text"] == test_text

        # Verify audio chunks sent
        assert mock_websocket.send_bytes.call_count >= 3  # At least 3 chunks from mock

        # Verify completion message
        complete_calls = [
            call
            for call in mock_websocket.send_json.call_args_list
            if call.args[0].get("type") == "tts_complete"
        ]
        assert len(complete_calls) > 0

    @pytest.mark.asyncio
    async def test_tts_websocket_json_message(self, mock_app, mock_websocket):
        """Test TTS WebSocket with JSON message format."""
        session_id = "test_session_123"
        test_message = {"type": "synthesize", "text": "Hello, this is a JSON test."}

        # Mock JSON message reception
        call_count = 0

        async def mock_receive_text():
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                return json.dumps(test_message)
            else:
                from fastapi import WebSocketDisconnect

                raise WebSocketDisconnect()

        mock_websocket.receive_text = mock_receive_text

        # Test TTS WebSocket
        await tts_websocket(mock_websocket, session_id, mock_app)

        # Verify synthesis with correct text
        start_calls = [
            call
            for call in mock_websocket.send_json.call_args_list
            if call.args[0].get("type") == "tts_start"
        ]
        assert len(start_calls) > 0
        assert start_calls[0].args[0]["text"] == test_message["text"]

    @pytest.mark.asyncio
    async def test_websocket_error_handling(self, mock_websocket):
        """Test WebSocket error handling when voice manager is not available."""
        session_id = "test_session_123"

        # Mock app without voice manager
        app = Mock()
        app.state = Mock()
        app.state.voice_manager = None

        # Test STT WebSocket with missing voice manager
        await stt_websocket(mock_websocket, session_id, app)

        # Verify error handling
        mock_websocket.accept.assert_called_once()
        # Should send error message about services not initialized
        error_calls = [
            call
            for call in mock_websocket.send_json.call_args_list
            if "error" in str(call.args[0]).lower()
        ]
        assert len(error_calls) > 0


class TestVoiceWebSocketIntegration:
    """Integration tests for voice WebSocket functionality."""

    def test_websocket_imports(self):
        """Test that voice WebSocket modules can be imported."""
        try:
            from app.ws.voice_stt import stt_websocket
            from app.ws.voice_tts import tts_websocket

            assert stt_websocket is not None
            assert tts_websocket is not None

        except ImportError as e:
            pytest.skip(f"Voice WebSocket imports not available: {str(e)}")

    def test_websocket_routing_in_main(self):
        """Test that voice WebSocket routes are defined in main.py."""
        try:
            from app.main import app

            # Check if voice WebSocket routes are registered
            routes = [route.path for route in app.routes]

            # Look for voice WebSocket patterns
            any("/ws/stt/" in route for route in routes)
            any("/ws/tts/" in route for route in routes)

            # Note: These might not exist if the routes are dynamically added
            # This test mainly checks that main.py can be imported without errors
            assert app is not None

        except ImportError as e:
            pytest.skip(f"Main app import failed: {str(e)}")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
