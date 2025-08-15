"""
Unit tests for voice services in the Improved Local AI Assistant.

Tests the VoiceManager, VoskSTTService, and PiperTTSService classes.
"""

from pathlib import Path
from unittest.mock import AsyncMock
from unittest.mock import Mock
from unittest.mock import patch

import pytest

# Import the voice services
try:
    from services.piper_tts_service import PiperTTSService
    from services.voice_manager import VoiceManager
    from services.vosk_stt_service import VoskSTTService

    VOICE_IMPORTS_AVAILABLE = True
except ImportError as e:
    VOICE_IMPORTS_AVAILABLE = False
    IMPORT_ERROR = str(e)


@pytest.mark.skipif(
    not VOICE_IMPORTS_AVAILABLE,
    reason=f"Voice imports not available: {IMPORT_ERROR if not VOICE_IMPORTS_AVAILABLE else ''}",
)
class TestVoiceManager:
    """Test cases for VoiceManager."""

    @pytest.fixture
    def voice_config(self):
        """Voice configuration for testing."""
        return {
            "voice": {
                "enabled": True,
                "stt": {"enabled": True, "model_name": "test-model", "sample_rate": 16000},
                "tts": {
                    "enabled": True,
                    "voice_name": "test-voice",
                    "sample_rate": 22050,
                    "speed": 1.0,
                },
            }
        }

    @pytest.fixture
    def mock_stt_service(self):
        """Mock STT service."""
        mock = Mock()
        mock.is_available.return_value = True
        mock.create_recognizer = AsyncMock(return_value=True)
        mock.destroy_recognizer = AsyncMock(return_value=True)
        mock.process_audio = AsyncMock(return_value={"partial": "test"})
        return mock

    @pytest.fixture
    def mock_tts_service(self):
        """Mock TTS service."""
        mock = Mock()
        mock.is_available.return_value = True

        async def mock_synthesize_stream(text):
            # Simulate streaming audio chunks
            for i in range(3):
                yield b"audio_chunk_" + str(i).encode()

        mock.synthesize_stream = mock_synthesize_stream
        return mock

    @patch("services.voice_manager.VoskSTTService")
    @patch("services.voice_manager.PiperTTSService")
    def test_voice_manager_initialization(self, mock_piper, mock_vosk, voice_config):
        """Test VoiceManager initialization."""
        # Setup mocks
        mock_vosk.return_value.is_available.return_value = True
        mock_piper.return_value.is_available.return_value = True

        # Create VoiceManager
        voice_manager = VoiceManager(voice_config)

        # Verify initialization
        assert voice_manager.config == voice_config
        assert voice_manager.stt_service is not None
        assert voice_manager.tts_service is not None
        assert voice_manager.voice_sessions == {}
        assert voice_manager.is_voice_available() == True

    @patch("services.voice_manager.VoskSTTService")
    @patch("services.voice_manager.PiperTTSService")
    @pytest.mark.asyncio
    async def test_voice_session_management(self, mock_piper, mock_vosk, voice_config):
        """Test voice session creation and destruction."""
        # Setup mocks
        mock_stt = Mock()
        mock_stt.create_recognizer = AsyncMock(return_value=True)
        mock_stt.destroy_recognizer = AsyncMock(return_value=True)
        mock_vosk.return_value = mock_stt
        mock_piper.return_value.is_available.return_value = True

        voice_manager = VoiceManager(voice_config)
        session_id = "test_session_123"

        # Test session creation
        success = await voice_manager.create_voice_session(session_id)
        assert success == True
        assert session_id in voice_manager.voice_sessions
        assert voice_manager.metrics["active_voice_sessions"] == 1

        # Test session state
        session_state = voice_manager.get_voice_session_state(session_id)
        assert session_state is not None
        assert session_state["is_listening"] == False
        assert session_state["is_speaking"] == False

        # Test session destruction
        success = await voice_manager.destroy_voice_session(session_id)
        assert success == True
        assert session_id not in voice_manager.voice_sessions
        assert voice_manager.metrics["active_voice_sessions"] == 0

    @patch("services.voice_manager.VoskSTTService")
    @patch("services.voice_manager.PiperTTSService")
    @pytest.mark.asyncio
    async def test_audio_processing(self, mock_piper, mock_vosk, voice_config):
        """Test audio chunk processing."""
        # Setup mocks
        mock_stt = Mock()
        mock_stt.create_recognizer = AsyncMock(return_value=True)
        mock_stt.process_audio = AsyncMock(return_value={"partial": "hello"})
        mock_vosk.return_value = mock_stt
        mock_piper.return_value.is_available.return_value = True

        voice_manager = VoiceManager(voice_config)
        session_id = "test_session_123"

        # Create session
        await voice_manager.create_voice_session(session_id)

        # Test audio processing
        audio_data = b"fake_audio_data"
        result = await voice_manager.process_audio_chunk(session_id, audio_data)

        assert result == {"partial": "hello"}
        mock_stt.process_audio.assert_called_once_with(session_id, audio_data)
        assert voice_manager.metrics["stt_requests"] == 1

    @patch("services.voice_manager.VoskSTTService")
    @patch("services.voice_manager.PiperTTSService")
    @pytest.mark.asyncio
    async def test_tts_synthesis(self, mock_piper, mock_vosk, voice_config):
        """Test text-to-speech synthesis."""
        # Setup mocks
        mock_vosk.return_value.is_available.return_value = True

        async def mock_synthesize_stream(text):
            for i in range(3):
                yield f"audio_chunk_{i}".encode()

        mock_tts = Mock()
        mock_tts.synthesize_stream = mock_synthesize_stream
        mock_piper.return_value = mock_tts

        voice_manager = VoiceManager(voice_config)
        session_id = "test_session_123"

        # Create session
        await voice_manager.create_voice_session(session_id)

        # Test TTS synthesis
        test_text = "Hello, this is a test."
        chunks = []

        async for chunk in voice_manager.synthesize_response(session_id, test_text):
            chunks.append(chunk)

        assert len(chunks) == 3
        assert chunks[0] == b"audio_chunk_0"
        assert voice_manager.metrics["tts_requests"] == 1

    @patch("services.voice_manager.VoskSTTService")
    @patch("services.voice_manager.PiperTTSService")
    def test_voice_metrics(self, mock_piper, mock_vosk, voice_config):
        """Test voice metrics collection."""
        mock_vosk.return_value.is_available.return_value = True
        mock_piper.return_value.is_available.return_value = True

        voice_manager = VoiceManager(voice_config)

        # Get initial metrics
        metrics = voice_manager.get_voice_metrics()

        assert "stt_requests" in metrics
        assert "tts_requests" in metrics
        assert "active_voice_sessions" in metrics
        assert "stt_available" in metrics
        assert "tts_available" in metrics
        assert metrics["stt_available"] == True
        assert metrics["tts_available"] == True


@pytest.mark.skipif(
    not VOICE_IMPORTS_AVAILABLE,
    reason=f"Voice imports not available: {IMPORT_ERROR if not VOICE_IMPORTS_AVAILABLE else ''}",
)
class TestVoskSTTService:
    """Test cases for VoskSTTService."""

    @pytest.fixture
    def stt_config(self):
        """STT configuration for testing."""
        return {"model_path": "/fake/model/path", "sample_rate": 16000}

    @patch("services.vosk_stt_service.vosk")
    def test_stt_service_initialization_no_model(self, mock_vosk, stt_config):
        """Test STT service initialization when model is not found."""
        mock_vosk.Model.side_effect = FileNotFoundError("Model not found")

        with pytest.raises(FileNotFoundError):
            VoskSTTService(stt_config)

    @patch("services.vosk_stt_service.vosk")
    @patch("services.vosk_stt_service.Path")
    def test_stt_service_initialization_success(self, mock_path, mock_vosk, stt_config):
        """Test successful STT service initialization."""
        # Mock path existence
        mock_path.return_value.exists.return_value = True
        mock_path.return_value.is_dir.return_value = True

        # Mock Vosk model
        mock_model = Mock()
        mock_vosk.Model.return_value = mock_model

        # Create service
        with patch.object(VoskSTTService, "_get_model_path", return_value=Path("/fake/path")):
            stt_service = VoskSTTService(stt_config)

        assert stt_service.model == mock_model
        assert stt_service.sample_rate == 16000
        assert stt_service.is_available() == True

    @patch("services.vosk_stt_service.vosk")
    @patch("services.vosk_stt_service.Path")
    @pytest.mark.asyncio
    async def test_recognizer_management(self, mock_path, mock_vosk, stt_config):
        """Test recognizer creation and destruction."""
        # Setup mocks
        mock_path.return_value.exists.return_value = True
        mock_path.return_value.is_dir.return_value = True
        mock_model = Mock()
        mock_recognizer = Mock()
        mock_vosk.Model.return_value = mock_model
        mock_vosk.KaldiRecognizer.return_value = mock_recognizer

        # Create service
        with patch.object(VoskSTTService, "_get_model_path", return_value=Path("/fake/path")):
            stt_service = VoskSTTService(stt_config)

        session_id = "test_session"

        # Test recognizer creation
        success = await stt_service.create_recognizer(session_id)
        assert success == True
        assert session_id in stt_service.recognizers

        # Test recognizer destruction
        success = await stt_service.destroy_recognizer(session_id)
        assert success == True
        assert session_id not in stt_service.recognizers


@pytest.mark.skipif(
    not VOICE_IMPORTS_AVAILABLE,
    reason=f"Voice imports not available: {IMPORT_ERROR if not VOICE_IMPORTS_AVAILABLE else ''}",
)
class TestPiperTTSService:
    """Test cases for PiperTTSService."""

    @pytest.fixture
    def tts_config(self):
        """TTS configuration for testing."""
        return {
            "voice_path": "/fake/voice/path.onnx",
            "sample_rate": 22050,
            "chunk_size": 1024,
            "speed": 1.0,
        }

    @patch("services.piper_tts_service.PiperVoice")
    @patch("services.piper_tts_service.Path")
    def test_tts_service_initialization_success(self, mock_path, mock_piper_voice, tts_config):
        """Test successful TTS service initialization."""
        # Mock path existence
        mock_path.return_value.exists.return_value = True

        # Mock Piper voice
        mock_voice = Mock()
        mock_piper_voice.load.return_value = mock_voice

        # Create service
        with patch.object(PiperTTSService, "_get_voice_path", return_value=Path("/fake/path.onnx")):
            tts_service = PiperTTSService(tts_config)

        assert tts_service.voice == mock_voice
        assert tts_service.sample_rate == 22050
        assert tts_service.is_available() == True

    @patch("services.piper_tts_service.PiperVoice")
    @patch("services.piper_tts_service.Path")
    @pytest.mark.asyncio
    async def test_tts_synthesis_stream(self, mock_path, mock_piper_voice, tts_config):
        """Test TTS synthesis streaming."""
        # Setup mocks
        mock_path.return_value.exists.return_value = True
        mock_voice = Mock()
        mock_piper_voice.load.return_value = mock_voice

        # Create service
        with patch.object(PiperTTSService, "_get_voice_path", return_value=Path("/fake/path.onnx")):
            tts_service = PiperTTSService(tts_config)

        # Mock synthesis
        with patch.object(tts_service, "_synthesize_audio", return_value=b"fake_audio_data"):
            chunks = []
            async for chunk in tts_service.synthesize_stream("Hello world"):
                chunks.append(chunk)

            assert len(chunks) > 0
            assert all(isinstance(chunk, bytes) for chunk in chunks)


class TestVoiceIntegration:
    """Integration tests for voice functionality."""

    @pytest.mark.asyncio
    async def test_voice_service_imports(self):
        """Test that voice services can be imported."""
        try:
            from services.piper_tts_service import PiperTTSService
            from services.voice_manager import VoiceManager
            from services.vosk_stt_service import VoskSTTService

            # Test that classes can be instantiated (will fail without models, but imports work)
            assert VoiceManager is not None
            assert VoskSTTService is not None
            assert PiperTTSService is not None

        except ImportError as e:
            pytest.skip(f"Voice dependencies not installed: {str(e)}")

    def test_voice_config_structure(self):
        """Test voice configuration structure."""
        from app.core import load_config

        config = load_config()
        voice_config = config.get("voice", {})

        # Test basic structure
        assert isinstance(voice_config, dict)

        if voice_config.get("enabled"):
            assert "stt" in voice_config
            assert "tts" in voice_config

            stt_config = voice_config["stt"]
            tts_config = voice_config["tts"]

            assert isinstance(stt_config, dict)
            assert isinstance(tts_config, dict)

            # Test required fields
            assert "enabled" in stt_config
            assert "sample_rate" in stt_config
            assert "enabled" in tts_config
            assert "sample_rate" in tts_config


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
