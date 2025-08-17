"""
Piper Text-to-Speech Service for the Improved Local AI Assistant.

This module provides the PiperTTSService class that handles offline speech
synthesis using the Piper TTS library.
"""

from typing import Optional
import asyncio
import logging
import wave
from collections.abc import AsyncGenerator
from pathlib import Path

try:
    from piper import PiperVoice

    piper = True  # Flag to indicate piper is available
except ImportError:
    piper = None
    PiperVoice = None


class PiperTTSService:
    """
    Text-to-speech service using Piper for offline synthesis.

    Provides streaming text-to-speech synthesis with configurable voices
    and audio quality settings.
    """

    def __init__(self, config: dict = None):
        """
        Initialize PiperTTSService with configuration.

        Args:
            config: Configuration dictionary
        """
        self.config = config or {}
        self.logger = logging.getLogger(__name__)

        # Check if Piper is available
        if piper is None or PiperVoice is None:
            raise ImportError("Piper TTS library not installed. Run: pip install piper-tts")

        # Voice management
        self.voice: Optional[PiperVoice] = None
        self.voice_path = self._get_voice_path()

        # Configuration
        self.sample_rate = self.config.get("sample_rate", 22050)
        self.chunk_size = self.config.get("chunk_size", 1024)  # Audio chunk size
        self.speed = self.config.get("speed", 1.0)  # Speech speed multiplier

        # Synthesis cancellation tracking
        self.active_synthesis: dict[str, bool] = {}  # session_id -> is_active

        # Initialize voice
        self._load_voice()

    def _get_voice_path(self) -> Path:
        """Get the path to the Piper voice model."""
        # Try configured path first
        if "voice_path" in self.config:
            voice_path = Path(self.config["voice_path"])
            if voice_path.exists():
                return voice_path

        # Try default voice name
        voice_name = self.config.get("voice_name", "en_US-lessac-medium")

        # Try default paths
        project_root = Path(__file__).parent.parent
        default_paths = [
            project_root / "models" / "piper" / voice_name / f"{voice_name}.onnx",
            project_root / "models" / "piper" / f"{voice_name}.onnx",
            Path.home() / ".cache" / "piper" / voice_name / f"{voice_name}.onnx",
        ]

        for path in default_paths:
            if path.exists():
                self.logger.info(f"Found Piper voice at: {path}")
                return path

        # Voice not found
        raise FileNotFoundError(
            f"Piper voice model not found. Please download a voice using:\n"
            f"python scripts/download_voice_models.py --piper {voice_name}\n"
            f"Searched paths: {[str(p) for p in default_paths]}"
        )

    def _load_voice(self):
        """Load the Piper voice model."""
        try:
            self.logger.info(f"Loading Piper voice from: {self.voice_path}")

            # Load voice model
            self.voice = PiperVoice.load(str(self.voice_path))

            # Configure voice settings
            if hasattr(self.voice, "config"):
                # Set sample rate if supported
                if hasattr(self.voice.config, "audio"):
                    self.sample_rate = getattr(
                        self.voice.config.audio, "sample_rate", self.sample_rate
                    )

            self.logger.info(f"Piper voice loaded successfully (sample_rate: {self.sample_rate})")

        except Exception as e:
            self.logger.error(f"Failed to load Piper voice: {str(e)}")
            raise

    async def synthesize_stream(
        self, text: str, session_id: str = None
    ) -> AsyncGenerator[bytes, None]:
        """
        Synthesize text to speech and stream audio chunks with cancellation support.

        Args:
            text: Text to synthesize
            session_id: Session identifier for cancellation tracking

        Yields:
            bytes: Audio chunks in PCM format
        """
        try:
            if not self.voice:
                raise RuntimeError("Piper voice not loaded")

            if not text.strip():
                self.logger.warning("Empty text provided for synthesis")
                return

            # Track active synthesis
            if session_id:
                self.active_synthesis[session_id] = True

            self.logger.debug(f"Synthesizing text: '{text[:50]}{'...' if len(text) > 50 else ''}'")

            # Use asyncio to run synthesis in thread pool to avoid blocking
            loop = asyncio.get_event_loop()

            # Synthesize audio
            audio_data = await loop.run_in_executor(None, self._synthesize_audio, text)

            # Stream audio in chunks with cancellation checking
            chunk_size = self.chunk_size * 2  # 2 bytes per sample (16-bit)

            for i in range(0, len(audio_data), chunk_size):
                # Check for cancellation
                if session_id and not self.active_synthesis.get(session_id, True):
                    self.logger.info(f"TTS synthesis cancelled for session {session_id}")
                    break

                chunk = audio_data[i : i + chunk_size]
                if chunk:
                    yield chunk

                    # Small delay to allow other tasks to run
                    await asyncio.sleep(0.001)

            # Clean up synthesis tracking
            if session_id:
                self.active_synthesis.pop(session_id, None)

            self.logger.debug(f"Finished streaming {len(audio_data)} bytes of audio")

        except Exception as e:
            # Clean up on error
            if session_id:
                self.active_synthesis.pop(session_id, None)
            self.logger.error(f"Failed to synthesize text: {str(e)}")
            raise

    async def cancel_synthesis(self, session_id: str) -> bool:
        """
        Cancel active synthesis for a session.

        Args:
            session_id: Session identifier

        Returns:
            bool: True if cancellation was successful
        """
        try:
            if session_id in self.active_synthesis:
                self.active_synthesis[session_id] = False
                self.logger.info(f"Cancelled TTS synthesis for session {session_id}")
                return True
            else:
                self.logger.debug(f"No active synthesis to cancel for session {session_id}")
                return False

        except Exception as e:
            self.logger.error(f"Failed to cancel synthesis for session {session_id}: {str(e)}")
            return False

    def _synthesize_audio(self, text: str) -> bytes:
        """
        Synthesize text to audio (blocking operation).

        Args:
            text: Text to synthesize

        Returns:
            bytes: PCM audio data
        """
        try:
            # Synthesize audio using Piper
            audio_chunks = []

            # Get audio chunks from Piper
            for audio_chunk in self.voice.synthesize(text):
                # audio_chunk.audio_int16_bytes contains the raw PCM data
                audio_chunks.append(audio_chunk.audio_int16_bytes)

            # Combine all audio chunks
            if audio_chunks:
                combined_audio = b"".join(audio_chunks)
                return combined_audio
            else:
                return b""

        except Exception as e:
            self.logger.error(f"Error in audio synthesis: {str(e)}")
            raise

    async def synthesize_to_file(self, text: str, output_path: Path) -> bool:
        """
        Synthesize text to an audio file.

        Args:
            text: Text to synthesize
            output_path: Output file path

        Returns:
            bool: True if synthesis successful
        """
        try:
            if not self.voice:
                raise RuntimeError("Piper voice not loaded")

            self.logger.info(f"Synthesizing text to file: {output_path}")

            # Ensure output directory exists
            output_path.parent.mkdir(parents=True, exist_ok=True)

            # Use asyncio to run synthesis in thread pool
            loop = asyncio.get_event_loop()

            await loop.run_in_executor(None, self._synthesize_to_file_blocking, text, output_path)

            self.logger.info(f"Successfully synthesized audio to: {output_path}")
            return True

        except Exception as e:
            self.logger.error(f"Failed to synthesize to file: {str(e)}")
            return False

    def _synthesize_to_file_blocking(self, text: str, output_path: Path):
        """
        Synthesize text to file (blocking operation).

        Args:
            text: Text to synthesize
            output_path: Output file path
        """
        with wave.open(str(output_path), "wb") as wav_file:
            wav_file.setnchannels(1)  # Mono
            wav_file.setsampwidth(2)  # 16-bit
            wav_file.setframerate(self.sample_rate)

            # Synthesize audio
            self.voice.synthesize(text, wav_file)

    def get_voice_info(self) -> dict:
        """
        Get information about the loaded voice.

        Returns:
            Dict: Voice information
        """
        if not self.voice:
            return {"available": False}

        info = {
            "available": True,
            "voice_path": str(self.voice_path),
            "sample_rate": self.sample_rate,
            "chunk_size": self.chunk_size,
            "speed": self.speed,
        }

        # Add voice-specific information if available
        if hasattr(self.voice, "config"):
            config = self.voice.config

            if hasattr(config, "model_name"):
                info["model_name"] = config.model_name

            if hasattr(config, "language"):
                info["language"] = config.language

            if hasattr(config, "dataset"):
                info["dataset"] = config.dataset

        return info

    def get_service_info(self) -> dict:
        """
        Get service information and status.

        Returns:
            Dict: Service information
        """
        return {
            "service": "PiperTTSService",
            "available": self.voice is not None,
            "voice_info": self.get_voice_info(),
            "config": {
                "sample_rate": self.sample_rate,
                "chunk_size": self.chunk_size,
                "speed": self.speed,
            },
        }

    def is_available(self) -> bool:
        """
        Check if the TTS service is available.

        Returns:
            bool: True if service is ready to use
        """
        return self.voice is not None

    def set_speed(self, speed: float):
        """
        Set the speech speed multiplier.

        Args:
            speed: Speed multiplier (0.5 = half speed, 2.0 = double speed)
        """
        self.speed = max(0.1, min(3.0, speed))  # Clamp between 0.1 and 3.0
        self.logger.info(f"Set TTS speed to: {self.speed}")

    def get_supported_formats(self) -> list:
        """
        Get supported audio formats.

        Returns:
            list: List of supported formats
        """
        return ["wav", "pcm"]
