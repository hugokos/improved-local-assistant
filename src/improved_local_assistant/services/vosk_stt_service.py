"""
Vosk Speech-to-Text Service for the Improved Local AI Assistant.

This module provides the VoskSTTService class that handles offline speech
recognition using the Vosk library.
"""

import json
import logging
from pathlib import Path
from typing import Dict
from typing import Optional

try:
    import vosk
except ImportError:
    vosk = None


class VoskSTTService:
    """
    Speech-to-text service using Vosk for offline recognition.

    Provides real-time speech recognition with partial and final results,
    supporting multiple concurrent sessions.
    """

    def __init__(self, config: Dict = None):
        """
        Initialize VoskSTTService with configuration.

        Args:
            config: Configuration dictionary
        """
        self.config = config or {}
        self.logger = logging.getLogger(__name__)

        # Check if Vosk is available
        if vosk is None:
            raise ImportError("Vosk library not installed. Run: pip install vosk")

        # Model and recognizer management
        self.model: Optional[vosk.Model] = None
        self.recognizers: Dict[str, vosk.KaldiRecognizer] = {}  # Free dictation recognizers
        self.command_recognizers: Dict[str, vosk.KaldiRecognizer] = {}  # Command recognizers

        # Configuration - Vosk works best at 16kHz
        self.sample_rate = self.config.get("sample_rate", 16000)  # Vosk standard rate
        self.model_path = self._get_model_path()

        # Command grammar for voice control
        self.command_phrases = [
            "stop",
            "cancel",
            "mute",
            "repeat",
            "slower",
            "slow down",
            "faster",
            "speed up",
            "normal speed",
            "reset speed",
            "new chat",
            "clear chat",
            "start over",
            "summarize",
            "summary",
            "cite sources",
            "show sources",
            "delete last",
            "undo",
        ]
        self.command_grammar = json.dumps(self.command_phrases)

        # Initialize model
        self._load_model()

    def _get_model_path(self) -> Path:
        """Get the path to the Vosk model."""
        # Try configured path first
        if "model_path" in self.config:
            model_path = Path(self.config["model_path"])
            if model_path.exists():
                return model_path

        # Try default paths
        project_root = Path(__file__).parent.parent
        default_paths = [
            project_root / "models" / "vosk" / "vosk-model-small-en",
            project_root / "models" / "vosk" / "vosk-model-en",
            Path.home() / ".cache" / "vosk" / "vosk-model-small-en-us-0.15",
        ]

        for path in default_paths:
            if path.exists() and path.is_dir():
                self.logger.info(f"Found Vosk model at: {path}")
                return path

        # Model not found
        raise FileNotFoundError(
            f"Vosk model not found. Please download a model using:\n"
            f"python scripts/download_voice_models.py --vosk small-en\n"
            f"Searched paths: {[str(p) for p in default_paths]}"
        )

    def _load_model(self):
        """Load the Vosk model."""
        try:
            self.logger.info(f"Loading Vosk model from: {self.model_path}")

            # Set log level to reduce Vosk output
            vosk.SetLogLevel(-1)

            # Load model
            self.model = vosk.Model(str(self.model_path))

            self.logger.info("Vosk model loaded successfully")

        except Exception as e:
            self.logger.error(f"Failed to load Vosk model: {str(e)}")
            raise

    async def create_recognizer(self, session_id: str) -> bool:
        """
        Create recognizers for a session (both free dictation and command).

        Args:
            session_id: Session identifier

        Returns:
            bool: True if recognizers created successfully
        """
        try:
            if not self.model:
                raise RuntimeError("Vosk model not loaded")

            if session_id in self.recognizers:
                self.logger.warning(f"Recognizers for session {session_id} already exist")
                return True

            # Create free dictation recognizer
            recognizer = vosk.KaldiRecognizer(self.model, self.sample_rate)
            recognizer.SetMaxAlternatives(1)  # Only return best result
            recognizer.SetWords(True)  # Include word-level timing
            self.recognizers[session_id] = recognizer

            # Create command recognizer with constrained grammar
            command_recognizer = vosk.KaldiRecognizer(
                self.model, self.sample_rate, self.command_grammar
            )
            command_recognizer.SetMaxAlternatives(1)
            command_recognizer.SetWords(True)
            self.command_recognizers[session_id] = command_recognizer

            self.logger.info(
                f"Created Vosk recognizers (dictation + command) for session: {session_id}"
            )
            return True

        except Exception as e:
            self.logger.error(f"Failed to create recognizers for session {session_id}: {str(e)}")
            return False

    async def destroy_recognizer(self, session_id: str) -> bool:
        """
        Destroy recognizers for a session.

        Args:
            session_id: Session identifier

        Returns:
            bool: True if recognizers destroyed successfully
        """
        try:
            if session_id in self.recognizers:
                del self.recognizers[session_id]
                self.logger.info(f"Destroyed Vosk dictation recognizer for session: {session_id}")

            if session_id in self.command_recognizers:
                del self.command_recognizers[session_id]
                self.logger.info(f"Destroyed Vosk command recognizer for session: {session_id}")

            return True

        except Exception as e:
            self.logger.error(f"Failed to destroy recognizers for session {session_id}: {str(e)}")
            return False

    async def finalize_utterance(self, session_id: str) -> Dict:
        """
        Finalize per utterance the Vosk Server way.

        During speech: AcceptWaveform(frame) → False, emit PartialResult() only when changed
        On trailing silence (hangover timer): call Result() once, send stt_final, then Reset()

        Args:
            session_id: Session identifier

        Returns:
            Dict: Final recognition result with endpoint marker
        """
        try:
            if session_id not in self.recognizers:
                return {"error": f"No recognizer found for session {session_id}"}

            rec = self.recognizers[session_id]

            # VOSK SERVER PATTERN: Call Result() once per utterance
            # No need to feed silence - VAD already detected end-of-speech
            result = json.loads(rec.Result())
            text = (result.get("text") or "").strip()
            words = result.get("words", [])

            # Server-side validation: check if text is meaningful
            word_count = len(text.split()) if text else 0
            is_meaningful = self._is_meaningful_text(text, word_count)

            self.logger.info(
                f"🏁 Vosk Result() for {session_id}: '{text}' ({word_count} words, meaningful={is_meaningful})"
            )

            # CRITICAL: Reset() recognizer after Result() to avoid state bleed
            rec.Reset()
            self.logger.debug(f"🔄 Reset recognizer for {session_id}")

            return {
                "final": text if is_meaningful else "",
                "words": words if is_meaningful else [],
                "endpoint": True,
                "confidence": result.get("confidence", 0.0),
                "word_count": word_count,
                "meaningful": is_meaningful,
                "source": "vosk_result",
            }

        except Exception as e:
            self.logger.error(f"Failed to finalize utterance for {session_id}: {e}")
            # Try to reset recognizer even on error
            try:
                if session_id in self.recognizers:
                    self.recognizers[session_id].Reset()
            except:
                pass
            return {"error": str(e)}

    async def _reset_recognizer_fresh(self, session_id: str):
        """
        Create a fresh recognizer to avoid state bleed between utterances.
        This prevents the "same partial over and over" issue.
        """
        try:
            if session_id in self.recognizers:
                # Create brand new recognizer instead of just Reset()
                recognizer = vosk.KaldiRecognizer(self.model, self.sample_rate)
                recognizer.SetMaxAlternatives(1)
                recognizer.SetWords(True)
                self.recognizers[session_id] = recognizer

                # Also reset command recognizer
                if session_id in self.command_recognizers:
                    command_recognizer = vosk.KaldiRecognizer(
                        self.model, self.sample_rate, self.command_grammar
                    )
                    command_recognizer.SetMaxAlternatives(1)
                    command_recognizer.SetWords(True)
                    self.command_recognizers[session_id] = command_recognizer

                self.logger.debug(f"🔄 Created fresh recognizers for {session_id}")

        except Exception as e:
            self.logger.error(f"Failed to reset recognizer for {session_id}: {e}")
            # Fallback to simple reset
            if session_id in self.recognizers:
                self.recognizers[session_id].Reset()
            if session_id in self.command_recognizers:
                self.command_recognizers[session_id].Reset()

    def _is_meaningful_text(self, text: str, word_count: int) -> bool:
        """
        Determine if transcribed text is meaningful enough to commit.

        Args:
            text: Transcribed text
            word_count: Number of words in text

        Returns:
            bool: True if text should be committed
        """
        if not text or word_count == 0:
            return False

        # Allow certain single words that are valid commands/responses
        valid_single_words = {
            "yes",
            "no",
            "okay",
            "ok",
            "hello",
            "hi",
            "help",
            "stop",
            "start",
            "thanks",
            "thank",
            "please",
            "sorry",
            "what",
            "why",
            "how",
            "when",
            "where",
        }

        if word_count == 1:
            return text.lower() in valid_single_words

        # For multi-word phrases, require at least 2 words and minimum length
        if word_count >= 2 and len(text) >= 4:
            return True

        return False

    def _pcm16_stats(self, audio_data: bytes) -> tuple:
        """Calculate PCM16 audio statistics for debugging."""
        import math
        import struct

        n = len(audio_data) // 2
        if n == 0:
            return (0, 0, 0)

        # Unpack as little-endian 16-bit signed integers
        samples = struct.unpack("<" + "h" * n, audio_data[: n * 2])
        absmax = max(abs(v) for v in samples)
        rms = int(math.sqrt(sum(v * v for v in samples) / n))
        return absmax, rms, n

    async def process_audio(self, session_id: str, audio_data: bytes) -> Dict:
        """
        Process audio data for speech recognition with proper VAD coupling.

        Args:
            session_id: Session identifier
            audio_data: PCM audio data (16-bit, 16kHz, exact frame timing)

        Returns:
            Dict: Recognition result with 'partial', 'final', or 'command' text
        """
        try:
            if session_id not in self.recognizers:
                raise ValueError(f"No recognizer found for session {session_id}")

            # Validate frame size for WebRTC VAD compatibility
            # WebRTC VAD requires exact 10/20/30ms frames at 16kHz
            expected_frame_sizes = [320, 640, 960]  # 10ms, 20ms, 30ms at 16kHz (samples * 2 bytes)
            if len(audio_data) not in expected_frame_sizes:
                # Log frame size issues occasionally
                if hash(audio_data) % 50 == 0:  # Log ~2% of frames
                    self.logger.debug(
                        f"⚠️ Non-standard frame size for {session_id}: {len(audio_data)} bytes (expected: {expected_frame_sizes})"
                    )

            # Audio amplitude debugging (log occasionally)
            if len(audio_data) > 0 and hash(audio_data) % 100 == 0:  # Log ~1% of chunks
                absmax, rms, n = self._pcm16_stats(audio_data)
                self.logger.debug(
                    f"🎵 Server audio stats for {session_id}: bytes={len(audio_data)} absmax={absmax} rms={rms} samples={n} rate={self.sample_rate}"
                )

            # Process with both recognizers in parallel
            dictation_recognizer = self.recognizers[session_id]
            command_recognizer = self.command_recognizers.get(session_id)

            # Check command recognizer first (higher priority)
            if command_recognizer:
                if command_recognizer.AcceptWaveform(audio_data):
                    cmd_result = json.loads(command_recognizer.Result())
                    cmd_text = (cmd_result.get("text") or "").strip()
                    # Check if the command text contains any of our command phrases
                    if cmd_text and any(
                        phrase in cmd_text.lower() for phrase in self.command_phrases
                    ):
                        self.logger.info(f"Voice command detected for {session_id}: '{cmd_text}'")
                        # Reset command recognizer for next command
                        command_recognizer.Reset()
                        return {
                            "command": cmd_text,
                            "endpoint": True,
                            "confidence": cmd_result.get("confidence", 0.0),
                        }

            # Process with dictation recognizer
            # KEY FIX: Only call Result() when VAD explicitly signals end-of-utterance
            # During normal processing, only use PartialResult()
            if dictation_recognizer.AcceptWaveform(audio_data):
                # This should rarely happen during normal speech
                # Only when Vosk's internal VAD detects a very long pause
                result = json.loads(dictation_recognizer.Result())
                text = (result.get("text") or "").strip()

                self.logger.info(f"Vosk internal VAD triggered final for {session_id}: '{text}'")
                return {
                    "final": text,
                    "endpoint": True,
                    "confidence": result.get("confidence", 0.0),
                    "words": result.get("words", []),
                    "source": "vosk_vad",
                }
            else:
                # Normal case: get partial result while speech is ongoing
                partial_result = json.loads(dictation_recognizer.PartialResult())
                partial_text = partial_result.get("partial", "").strip()

                # VOSK SERVER PATTERN: Only emit partials when they change
                # Track last partial to avoid spam
                if not hasattr(self, "_last_partials"):
                    self._last_partials = {}

                last_partial = self._last_partials.get(session_id, "")

                # Also check command recognizer for partial commands
                if command_recognizer:
                    cmd_partial = json.loads(command_recognizer.PartialResult())
                    cmd_partial_text = cmd_partial.get("partial", "").strip()

                    # If command partial contains any of our command phrases, use it
                    if cmd_partial_text and any(
                        phrase in cmd_partial_text.lower() for phrase in self.command_phrases
                    ):
                        if cmd_partial_text != last_partial:
                            self._last_partials[session_id] = cmd_partial_text
                            self.logger.debug(
                                f"Partial command for {session_id}: '{cmd_partial_text}'"
                            )
                            return {"partial_command": cmd_partial_text}
                        return {}  # Same command partial, don't emit

                # Only emit partial if it changed
                if partial_text and partial_text != last_partial:
                    self._last_partials[session_id] = partial_text
                    self.logger.debug(f"Partial STT result for {session_id}: '{partial_text}'")
                    return {"partial": partial_text}
                else:
                    # No change in partial text
                    return {}

        except Exception as e:
            self.logger.error(f"Failed to process audio for session {session_id}: {str(e)}")
            return {"error": str(e)}

    def reset_recognizer(self, session_id: str) -> bool:
        """
        Reset a recognizer to clear its state.

        Args:
            session_id: Session identifier

        Returns:
            bool: True if reset successfully
        """
        try:
            if session_id not in self.recognizers:
                return False

            recognizer = self.recognizers[session_id]

            # Reset recognizer state
            recognizer.Reset()

            self.logger.debug(f"Reset recognizer for session: {session_id}")
            return True

        except Exception as e:
            self.logger.error(f"Failed to reset recognizer for session {session_id}: {str(e)}")
            return False

    def get_recognizer_info(self, session_id: str) -> Optional[Dict]:
        """
        Get information about a recognizer.

        Args:
            session_id: Session identifier

        Returns:
            Dict: Recognizer information or None if not found
        """
        if session_id not in self.recognizers:
            return None

        return {
            "session_id": session_id,
            "sample_rate": self.sample_rate,
            "model_path": str(self.model_path),
            "active": True,
        }

    def get_service_info(self) -> Dict:
        """
        Get service information and status.

        Returns:
            Dict: Service information
        """
        return {
            "service": "VoskSTTService",
            "available": self.model is not None,
            "model_path": str(self.model_path) if self.model_path else None,
            "sample_rate": self.sample_rate,
            "active_recognizers": len(self.recognizers),
            "sessions": list(self.recognizers.keys()),
        }

    def is_available(self) -> bool:
        """
        Check if the STT service is available.

        Returns:
            bool: True if service is ready to use
        """
        return self.model is not None
