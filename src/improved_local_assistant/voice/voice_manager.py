"""
Voice Manager for handling voice processing operations.

This module provides the VoiceManager class that coordinates speech-to-text
and text-to-speech operations using Vosk and Piper respectively.
"""

import asyncio
import logging
import time
from datetime import datetime
from typing import AsyncGenerator
from typing import Dict
from typing import Optional

from .piper_tts_service import PiperTTSService
from .vosk_stt_service import VoskSTTService
from .webrtc_vad_service import WebRTCVADService


class VoiceManager:
    """
    Manages voice processing operations including STT and TTS.

    Coordinates between speech recognition and synthesis services,
    maintains voice session state, and provides metrics.
    """

    def __init__(self, config: Dict = None):
        """
        Initialize VoiceManager with configuration.

        Args:
            config: Configuration dictionary
        """
        self.config = config or {}
        self.logger = logging.getLogger(__name__)

        # Voice services
        self.stt_service: Optional[VoskSTTService] = None
        self.tts_service: Optional[PiperTTSService] = None
        self.vad_service: Optional[WebRTCVADService] = None

        # Voice session management
        self.voice_sessions: Dict[str, Dict] = {}

        # HALF-DUPLEX CONTROL: Proven pattern from Home Assistant/Wyoming
        self.half_duplex_sessions: Dict[str, Dict] = {}  # session_id -> half_duplex_state

        # Performance metrics
        self.metrics = {
            "stt_requests": 0,
            "tts_requests": 0,
            "avg_stt_latency": 0.0,
            "avg_tts_latency": 0.0,
            "active_voice_sessions": 0,
            "total_audio_processed": 0,
            "errors": 0,
        }

        # Initialize services
        self._initialize_services()

    def _initialize_services(self):
        """Initialize STT and TTS services."""
        try:
            voice_config = self.config.get("voice", {})

            # Initialize STT service
            stt_config = voice_config.get("stt", {})
            if stt_config.get("enabled", True):
                self.stt_service = VoskSTTService(stt_config)
                self.logger.info("STT service initialized")

            # Initialize TTS service
            tts_config = voice_config.get("tts", {})
            if tts_config.get("enabled", True):
                self.tts_service = PiperTTSService(tts_config)
                self.logger.info("TTS service initialized")

            # Initialize VAD service
            vad_config = voice_config.get("vad", {})
            if vad_config.get("enabled", True) and WebRTCVADService.is_available():
                try:
                    self.vad_service = WebRTCVADService(vad_config)
                    self.logger.info("WebRTC VAD service initialized")
                except Exception as e:
                    self.logger.warning(f"Failed to initialize WebRTC VAD: {str(e)}")
                    self.vad_service = None
            else:
                self.logger.info("WebRTC VAD service disabled or not available")

        except Exception as e:
            self.logger.error(f"Failed to initialize voice services: {str(e)}")
            raise

    async def create_voice_session(self, session_id: str) -> bool:
        """
        Create a new voice session.

        Args:
            session_id: Session identifier

        Returns:
            bool: True if session created successfully
        """
        try:
            if session_id in self.voice_sessions:
                self.logger.warning(f"Voice session {session_id} already exists")
                return True

            # Create session state
            self.voice_sessions[session_id] = {
                "created_at": datetime.now(),
                "is_listening": False,
                "is_speaking": False,
                "stt_active": False,
                "tts_active": False,
                "audio_level": 0.0,
                "last_transcript": "",
                "partial_transcript": "",
                "metrics": {
                    "stt_requests": 0,
                    "tts_requests": 0,
                    "audio_chunks_processed": 0,
                    "total_audio_duration": 0.0,
                },
            }

            # HALF-DUPLEX STATE: Strict speaking/listening control
            self.half_duplex_sessions[session_id] = {
                "mode": "listening",  # listening, speaking, hold_off
                "mic_muted": False,
                "tts_playing": False,
                "hold_off_until": 0,  # timestamp when hold-off ends
                "hold_off_duration_ms": 250,  # 250ms post-speech hold-off
                "barge_in_enabled": True,
                "vad_state": {
                    "is_active": False,
                    "voiced_frames": 0,
                    "silence_frames": 0,
                    "hangover_ms": 400,  # 400ms hangover after last voiced frame
                    "preroll_ms": 200,  # 200ms preroll buffer
                    "preroll_buffer": [],
                    "energy_baseline": 0.0,
                    "energy_threshold": 0.02,  # RMS threshold above baseline
                    "frames_to_start": 3,  # 3 voiced frames to start utterance
                    "frames_to_end": 5,  # 5 silence frames to end (with hangover)
                },
            }

            # Initialize STT recognizer for this session
            if self.stt_service:
                await self.stt_service.create_recognizer(session_id)

            self.metrics["active_voice_sessions"] += 1
            self.logger.info(f"Created voice session: {session_id}")

            return True

        except Exception as e:
            self.logger.error(f"Failed to create voice session {session_id}: {str(e)}")
            self.metrics["errors"] += 1
            return False

    def _set_half_duplex_mode(self, session_id: str, mode: str):
        """
        Set half-duplex mode following Home Assistant/Wyoming pattern.

        Args:
            session_id: Session identifier
            mode: 'listening', 'speaking', or 'hold_off'
        """
        if session_id not in self.half_duplex_sessions:
            return

        hd_state = self.half_duplex_sessions[session_id]
        old_mode = hd_state["mode"]
        hd_state["mode"] = mode

        if mode == "speaking":
            # MUTE MIC: Stop listening while speaking to avoid self-hearing
            hd_state["mic_muted"] = True
            hd_state["tts_playing"] = True
            self.logger.info(f"🔇 Half-duplex: {session_id} SPEAKING (mic muted)")

        elif mode == "listening":
            # UNMUTE MIC: Ready to listen
            hd_state["mic_muted"] = False
            hd_state["tts_playing"] = False
            hd_state["hold_off_until"] = 0
            self.logger.info(f"🎤 Half-duplex: {session_id} LISTENING (mic active)")

        elif mode == "hold_off":
            # POST-SPEECH HOLD-OFF: Brief pause before re-enabling mic
            hold_off_ms = hd_state["hold_off_duration_ms"]
            hd_state["hold_off_until"] = time.time() * 1000 + hold_off_ms
            hd_state["mic_muted"] = True
            hd_state["tts_playing"] = False
            self.logger.info(f"⏸️ Half-duplex: {session_id} HOLD_OFF ({hold_off_ms}ms)")

            # Schedule transition to listening
            asyncio.create_task(self._schedule_hold_off_end(session_id, hold_off_ms))

    async def _schedule_hold_off_end(self, session_id: str, hold_off_ms: int):
        """Schedule end of hold-off period."""
        await asyncio.sleep(hold_off_ms / 1000.0)
        if session_id in self.half_duplex_sessions:
            hd_state = self.half_duplex_sessions[session_id]
            if hd_state["mode"] == "hold_off":
                self._set_half_duplex_mode(session_id, "listening")

    def _should_process_audio(self, session_id: str) -> bool:
        """
        Check if audio should be processed based on half-duplex state.

        Returns:
            bool: True if audio should be processed
        """
        if session_id not in self.half_duplex_sessions:
            return True

        hd_state = self.half_duplex_sessions[session_id]

        # Don't process if mic is muted
        if hd_state["mic_muted"]:
            return False

        # Don't process during hold-off
        if hd_state["mode"] == "hold_off":
            current_time = time.time() * 1000
            if current_time < hd_state["hold_off_until"]:
                return False
            else:
                # Hold-off expired, transition to listening
                self._set_half_duplex_mode(session_id, "listening")

        # Only process in listening mode
        return hd_state["mode"] == "listening"

    async def destroy_voice_session(self, session_id: str) -> bool:
        """
        Destroy a voice session and clean up resources.

        Args:
            session_id: Session identifier

        Returns:
            bool: True if session destroyed successfully
        """
        try:
            if session_id not in self.voice_sessions:
                self.logger.warning(f"Voice session {session_id} not found")
                return True

            # Clean up STT recognizer
            if self.stt_service:
                await self.stt_service.destroy_recognizer(session_id)

            # Remove session
            del self.voice_sessions[session_id]

            # Clean up half-duplex state
            if session_id in self.half_duplex_sessions:
                del self.half_duplex_sessions[session_id]

            self.metrics["active_voice_sessions"] -= 1

            self.logger.info(f"Destroyed voice session: {session_id}")
            return True

        except Exception as e:
            self.logger.error(f"Failed to destroy voice session {session_id}: {str(e)}")
            self.metrics["errors"] += 1
            return False

    async def on_vad_end(self, session_id: str) -> Dict:
        """
        Called when VAD says the user stopped talking.

        Args:
            session_id: Session identifier

        Returns:
            Dict: Final recognition result with endpoint marker
        """
        try:
            if not self.stt_service:
                return {"error": "STT not available"}

            # Longer linger to let complete utterances finish (400ms)
            await asyncio.sleep(0.4)

            final = await self.stt_service.finalize_utterance(session_id)
            session = self.voice_sessions.get(session_id)
            if session is None:
                return {"error": f"Session {session_id} not found"}

            # KEY FIX: Merge last partial into final if Vosk returns empty final
            final_text = final.get("final", "").strip()
            if not final_text and session.get("partial_transcript"):
                final_text = session["partial_transcript"].strip()
                final["final"] = final_text
                self.logger.info(
                    f"Merged partial into empty final for {session_id}: '{final_text}'"
                )

            # Fragment prevention: Only drop very short fragments (1-2 chars like "a", "I")
            if final_text and len(final_text) < 3:
                self.logger.info(f"Dropping short fragment for {session_id}: '{final_text}'")
                final["final"] = ""  # Clear short fragments

            # Update session/cache - don't clear partial until after merge
            if "final" in final:
                session["last_transcript"] = final.get("final", "")
                session["partial_transcript"] = ""  # Clear after merge
                session["metrics"]["stt_requests"] += 1

            self.logger.info(
                f"VAD end for {session_id}: '{final.get('final', '')}' (endpoint={final.get('endpoint', False)})"
            )
            return final

        except Exception as e:
            self.logger.error(f"Failed to handle VAD end for session {session_id}: {str(e)}")
            self.metrics["errors"] += 1
            return {"error": str(e)}

    def _validate_vad_frame(self, frame_data: bytes) -> bool:
        """
        BULLETPROOF VAD: Validate frame meets WebRTC VAD requirements.

        WebRTC VAD requires exact 10/20/30ms frames at 16kHz:
        - 10ms: 160 samples = 320 bytes
        - 20ms: 320 samples = 640 bytes
        - 30ms: 480 samples = 960 bytes

        Args:
            frame_data: PCM16 audio frame

        Returns:
            bool: True if frame is valid for WebRTC VAD
        """
        valid_sizes = {320, 640, 960}  # 10ms, 20ms, 30ms at 16kHz

        if len(frame_data) not in valid_sizes:
            if len(frame_data) > 0:  # Only log non-empty invalid frames
                self.logger.debug(
                    f"❌ Invalid VAD frame size: {len(frame_data)} bytes (expected: {valid_sizes})"
                )
            return False

        # Additional validation: ensure even byte count (16-bit samples)
        if len(frame_data) % 2 != 0:
            self.logger.warning(
                f"❌ Odd frame size: {len(frame_data)} bytes (16-bit PCM requires even bytes)"
            )
            return False

        return True

    def _update_vad_hysteresis(self, session_id: str, is_speech: bool, frame_data: bytes) -> Dict:
        """
        Update VAD state with hysteresis and energy gating.

        Implements proven VAD patterns:
        - Energy gate + WebRTC VAD combination
        - Preroll buffer for first syllables
        - Hangover for natural speech endings
        - Hysteresis to prevent flutter

        Args:
            session_id: Session identifier
            is_speech: WebRTC VAD result
            frame_data: Audio frame for energy calculation

        Returns:
            Dict: VAD state update
        """
        if session_id not in self.half_duplex_sessions:
            return {"error": "Session not found"}

        hd_state = self.half_duplex_sessions[session_id]
        vad_state = hd_state["vad_state"]

        # Calculate RMS energy for energy gating
        import math
        import struct

        try:
            samples = struct.unpack(f"<{len(frame_data)//2}h", frame_data)
            rms = (
                math.sqrt(sum(s * s for s in samples) / len(samples)) / 32768.0 if samples else 0.0
            )

            # Update energy baseline (slow adaptation)
            if vad_state["energy_baseline"] == 0.0:
                vad_state["energy_baseline"] = rms
            else:
                # Slow adaptation: 99% old + 1% new
                vad_state["energy_baseline"] = vad_state["energy_baseline"] * 0.99 + rms * 0.01

            # Energy gate: RMS must be above baseline + threshold
            energy_gate = rms > (vad_state["energy_baseline"] + vad_state["energy_threshold"])

            # Combined decision: WebRTC VAD AND energy gate
            is_voiced = is_speech and energy_gate

        except Exception as e:
            self.logger.error(f"Energy calculation error: {e}")
            is_voiced = is_speech  # Fallback to WebRTC VAD only
            rms = 0.0

        # Update frame counters with hysteresis
        if is_voiced:
            vad_state["voiced_frames"] += 1
            vad_state["silence_frames"] = 0

            # Add to preroll buffer
            vad_state["preroll_buffer"].append(frame_data)
            if len(vad_state["preroll_buffer"]) > 10:  # Keep last 10 frames (~200ms at 20ms frames)
                vad_state["preroll_buffer"].pop(0)

            # Start utterance if we have enough voiced frames
            if (
                not vad_state["is_active"]
                and vad_state["voiced_frames"] >= vad_state["frames_to_start"]
            ):
                vad_state["is_active"] = True
                self.logger.info(
                    f"🎤 VAD: Utterance STARTED for {session_id} (voiced_frames={vad_state['voiced_frames']}, rms={rms:.4f})"
                )

                return {
                    "utterance_started": True,
                    "preroll_frames": vad_state["preroll_buffer"].copy(),
                    "voiced_frames": vad_state["voiced_frames"],
                    "rms": rms,
                    "energy_baseline": vad_state["energy_baseline"],
                }
        else:
            vad_state["silence_frames"] += 1
            vad_state["voiced_frames"] = max(
                0, vad_state["voiced_frames"] - 1
            )  # Decay voiced frames

            # End utterance if we have enough silence frames AND hangover expired
            if vad_state["is_active"] and vad_state["silence_frames"] >= vad_state["frames_to_end"]:
                # Calculate hangover time
                hangover_frames = int(vad_state["hangover_ms"] / 20)  # Assuming 20ms frames

                if vad_state["silence_frames"] >= hangover_frames:
                    vad_state["is_active"] = False
                    vad_state["voiced_frames"] = 0
                    vad_state["silence_frames"] = 0

                    self.logger.info(
                        f"🔇 VAD: Utterance ENDED for {session_id} (hangover={vad_state['hangover_ms']}ms, rms={rms:.4f})"
                    )

                    return {
                        "utterance_ended": True,
                        "hangover_ms": vad_state["hangover_ms"],
                        "total_voiced_frames": vad_state["voiced_frames"],
                        "rms": rms,
                    }

        return {
            "is_active": vad_state["is_active"],
            "voiced_frames": vad_state["voiced_frames"],
            "silence_frames": vad_state["silence_frames"],
            "rms": rms,
            "energy_gate": energy_gate if "energy_gate" in locals() else False,
        }

    async def process_vad_frame(self, session_id: str, frame_data: bytes) -> Dict:
        """
        BULLETPROOF VAD: Process frame with strict validation and hysteresis.

        Args:
            session_id: Session identifier
            frame_data: PCM audio frame (16-bit, 16kHz, exact timing)

        Returns:
            Dict: VAD result with speech detection info
        """
        try:
            if session_id not in self.voice_sessions:
                raise ValueError(f"Voice session {session_id} not found")

            # CRITICAL: Validate frame before processing
            if not self._validate_vad_frame(frame_data):
                return {"error": "Invalid VAD frame", "frame_size": len(frame_data)}

            # Check half-duplex state - don't process if mic is muted
            if not self._should_process_audio(session_id):
                return {
                    "skipped": "mic_muted_or_hold_off",
                    "mode": self.half_duplex_sessions[session_id]["mode"],
                }

            if not self.vad_service:
                # Fallback to simple RMS-based VAD
                return self._simple_vad(frame_data)

            # Process with WebRTC VAD (aggressiveness=2, not 3)
            vad_results = self.vad_service.process_audio(frame_data)

            if vad_results:
                # Get the last frame result
                is_speech, _ = vad_results[-1]

                # Apply hysteresis and energy gating
                hysteresis_result = self._update_vad_hysteresis(session_id, is_speech, frame_data)

                # Get WebRTC VAD state
                vad_state = self.vad_service.get_vad_state()

                return {
                    "is_speech": is_speech,
                    "is_speech_active": vad_state["is_speech_active"],
                    "speech_frames": vad_state["speech_frames"],
                    "silence_frames": vad_state["silence_frames"],
                    "vad_type": "webrtc_with_hysteresis",
                    "hysteresis": hysteresis_result,
                    "frame_size": len(frame_data),
                }
            else:
                return {"is_speech": False, "vad_type": "webrtc", "no_frames": True}

        except Exception as e:
            self.logger.error(f"Failed to process VAD frame for session {session_id}: {str(e)}")
            return {"error": str(e), "vad_type": "error"}

    def _simple_vad(self, frame_data: bytes) -> Dict:
        """Simple RMS-based VAD fallback."""
        import math
        import struct

        try:
            # Convert to samples
            samples = struct.unpack(f"<{len(frame_data)//2}h", frame_data)

            # Calculate RMS
            rms = math.sqrt(sum(s * s for s in samples) / len(samples)) if samples else 0

            # Simple threshold
            is_speech = rms > 500  # Adjust threshold as needed

            return {"is_speech": is_speech, "rms": rms, "vad_type": "simple"}

        except Exception as e:
            self.logger.error(f"Simple VAD error: {str(e)}")
            return {"is_speech": False, "vad_type": "simple", "error": str(e)}

    async def process_audio_chunk(self, session_id: str, audio_data: bytes) -> Dict:
        """
        Process audio chunk with half-duplex control and frame validation.

        Args:
            session_id: Session identifier
            audio_data: PCM audio data (16-bit, 16kHz)

        Returns:
            Dict: Recognition result with 'partial' or 'final' text
        """
        start_time = time.time()

        try:
            if session_id not in self.voice_sessions:
                raise ValueError(f"Voice session {session_id} not found")

            if not self.stt_service:
                raise RuntimeError("STT service not available")

            # HALF-DUPLEX CHECK: Don't process audio if mic is muted or in hold-off
            if not self._should_process_audio(session_id):
                hd_state = self.half_duplex_sessions.get(session_id, {})
                return {
                    "skipped": "half_duplex_blocked",
                    "mode": hd_state.get("mode", "unknown"),
                    "mic_muted": hd_state.get("mic_muted", False),
                }

            session = self.voice_sessions[session_id]
            session["stt_active"] = True
            session["metrics"]["audio_chunks_processed"] += 1

            # Process audio with STT service (streams 10-20ms frames)
            result = await self.stt_service.process_audio(session_id, audio_data)

            # Update session state
            if result.get("partial"):
                session["partial_transcript"] = result["partial"]
            elif result.get("final"):
                session["last_transcript"] = result["final"]
                session["partial_transcript"] = ""
                session["metrics"]["stt_requests"] += 1

            # Update metrics
            latency = time.time() - start_time
            self.metrics["stt_requests"] += 1
            self.metrics["total_audio_processed"] += len(audio_data)

            # Update rolling average latency
            if self.metrics["stt_requests"] > 1:
                self.metrics["avg_stt_latency"] = (
                    self.metrics["avg_stt_latency"] * (self.metrics["stt_requests"] - 1) + latency
                ) / self.metrics["stt_requests"]
            else:
                self.metrics["avg_stt_latency"] = latency

            session["stt_active"] = False
            return result

        except Exception as e:
            self.logger.error(f"Failed to process audio chunk for session {session_id}: {str(e)}")
            self.metrics["errors"] += 1

            if session_id in self.voice_sessions:
                self.voice_sessions[session_id]["stt_active"] = False

            return {"error": str(e)}

    async def synthesize_response(self, session_id: str, text: str) -> AsyncGenerator[bytes, None]:
        """
        Synthesize with half-duplex control and streaming like Wyoming/Piper.

        Args:
            session_id: Session identifier
            text: Text to synthesize

        Yields:
            bytes: Audio chunks (PCM format)
        """
        start_time = time.time()

        try:
            if session_id not in self.voice_sessions:
                self.logger.error(
                    f"Voice session {session_id} not found. Available sessions: {list(self.voice_sessions.keys())}"
                )
                raise ValueError(f"Voice session {session_id} not found")

            if not self.tts_service:
                raise RuntimeError("TTS service not available")

            session = self.voice_sessions[session_id]
            session["tts_active"] = True
            session["is_speaking"] = True
            session["barge_in_requested"] = False  # Reset barge-in flag

            # HALF-DUPLEX: Switch to speaking mode (mutes mic)
            self._set_half_duplex_mode(session_id, "speaking")

            # Stream audio chunks with barge-in checking
            chunk_count = 0
            first_chunk_time = None

            self.logger.debug(
                f"Starting TTS synthesis for session {session_id}, text: '{text[:50]}...'"
            )
            async for audio_chunk in self.tts_service.synthesize_stream(text, session_id):
                # Record TTFT (Time To First Token) for metrics
                if chunk_count == 0:
                    first_chunk_time = time.time()
                    ttft = first_chunk_time - start_time
                    self.logger.info(f"🔊 TTS TTFT: {ttft*1000:.1f}ms for session {session_id}")

                # Check for barge-in before yielding each chunk
                if session.get("barge_in_requested", False):
                    self.logger.info(
                        f"🚨 Barge-in detected for session {session_id}, stopping TTS after {chunk_count} chunks"
                    )
                    # HALF-DUPLEX: Immediate switch to listening on barge-in
                    self._set_half_duplex_mode(session_id, "listening")
                    break

                chunk_count += 1
                yield audio_chunk

            # TTS completed normally - enter hold-off period
            if not session.get("barge_in_requested", False):
                self.logger.info(
                    f"✅ TTS completed: {len(text)} chars, {chunk_count} chunks for session {session_id}"
                )
                # HALF-DUPLEX: Enter hold-off before re-enabling mic
                self._set_half_duplex_mode(session_id, "hold_off")

            # Update metrics
            total_latency = time.time() - start_time
            session["metrics"]["tts_requests"] += 1
            self.metrics["tts_requests"] += 1

            # Track TTFT separately
            if first_chunk_time:
                ttft = first_chunk_time - start_time
                session["metrics"]["last_ttft"] = ttft

            # Update rolling average latency
            if self.metrics["tts_requests"] > 1:
                self.metrics["avg_tts_latency"] = (
                    self.metrics["avg_tts_latency"] * (self.metrics["tts_requests"] - 1)
                    + total_latency
                ) / self.metrics["tts_requests"]
            else:
                self.metrics["avg_tts_latency"] = total_latency

            session["tts_active"] = False
            session["is_speaking"] = False

        except Exception as e:
            self.logger.error(f"❌ Failed to synthesize response for session {session_id}: {str(e)}")
            self.metrics["errors"] += 1

            if session_id in self.voice_sessions:
                session = self.voice_sessions[session_id]
                session["tts_active"] = False
                session["is_speaking"] = False

            # HALF-DUPLEX: Return to listening on error
            self._set_half_duplex_mode(session_id, "listening")

    def get_voice_session_state(self, session_id: str) -> Optional[Dict]:
        """
        Get the current state of a voice session.

        Args:
            session_id: Session identifier

        Returns:
            Dict: Session state or None if not found
        """
        return self.voice_sessions.get(session_id)

    def set_voice_session_listening(self, session_id: str, listening: bool):
        """
        Set the listening state for a voice session.

        Args:
            session_id: Session identifier
            listening: Whether the session is listening
        """
        if session_id in self.voice_sessions:
            self.voice_sessions[session_id]["is_listening"] = listening

    def update_audio_level(self, session_id: str, level: float):
        """
        Update the audio level for a voice session.

        Args:
            session_id: Session identifier
            level: Audio level (0.0 to 1.0)
        """
        if session_id in self.voice_sessions:
            self.voice_sessions[session_id]["audio_level"] = max(0.0, min(1.0, level))

    def get_voice_metrics(self) -> Dict:
        """
        Get voice processing metrics.

        Returns:
            Dict: Current metrics
        """
        return {
            **self.metrics,
            "stt_available": self.stt_service is not None,
            "tts_available": self.tts_service is not None,
            "active_sessions": list(self.voice_sessions.keys()),
            "timestamp": datetime.now().isoformat(),
        }

    def is_voice_available(self) -> bool:
        """
        Check if voice functionality is available.

        Returns:
            bool: True if both STT and TTS are available
        """
        return self.stt_service is not None and self.tts_service is not None

    async def handle_barge_in(self, session_id: str) -> bool:
        """
        Handle barge-in request - immediately stop TTS and switch to listening.

        This is CRITICAL for natural conversation flow. When user starts speaking
        while bot is talking, we must stop TTS immediately and start listening.

        Args:
            session_id: Session identifier

        Returns:
            bool: True if barge-in was handled successfully
        """
        try:
            if session_id not in self.voice_sessions:
                self.logger.warning(f"Barge-in requested for unknown session: {session_id}")
                return False

            session = self.voice_sessions[session_id]

            self.logger.info(f"🚨 BARGE-IN: User interrupted TTS for session {session_id}")

            # IMMEDIATE: Set barge-in flag to stop TTS streaming
            session["barge_in_requested"] = True
            session["is_speaking"] = False
            session["is_listening"] = True

            # IMMEDIATE: Cancel any active TTS synthesis
            if self.tts_service and session.get("tts_active", False):
                cancel_success = await self.tts_service.cancel_synthesis(session_id)
                session["tts_active"] = False
                self.logger.info(
                    f"TTS cancellation for {session_id}: {'success' if cancel_success else 'failed'}"
                )

            # Reset STT state for new utterance
            if self.stt_service and session_id in self.stt_service.recognizers:
                # Create fresh recognizer to avoid state contamination
                await self.stt_service._reset_recognizer_fresh(session_id)
                self.logger.info(f"Reset STT recognizer for barge-in session {session_id}")

            # Clear any partial transcripts from interrupted session
            session["partial_transcript"] = ""
            session["last_transcript"] = ""

            self.logger.info(f"✅ Barge-in handled successfully for session {session_id}")
            return True

        except Exception as e:
            self.logger.error(f"❌ Failed to handle barge-in for session {session_id}: {str(e)}")
            self.metrics["errors"] += 1
            return False

    async def process_voice_command(self, session_id: str, command: str) -> Dict:
        """
        Process voice command (not sent to LLM).

        Args:
            session_id: Session identifier
            command: Voice command text

        Returns:
            Dict: Command processing result
        """
        try:
            command = command.lower().strip()
            self.logger.info(f"Processing voice command for session {session_id}: '{command}'")

            # Map commands to actions with fuzzy matching for compound commands
            if command in ["stop", "cancel", "mute"]:
                await self.handle_barge_in(session_id)
                return {"action": "stop_tts", "success": True}

            elif command == "repeat":
                return {"action": "repeat_last", "success": True}

            elif any(phrase in command for phrase in ["slower", "slow down"]):
                if self.tts_service:
                    self.tts_service.set_speed(0.8)
                return {"action": "speed_change", "speed": 0.8, "success": True}

            elif any(phrase in command for phrase in ["faster", "speed up"]):
                if self.tts_service:
                    self.tts_service.set_speed(1.2)
                return {"action": "speed_change", "speed": 1.2, "success": True}

            elif any(phrase in command for phrase in ["normal speed", "reset speed"]):
                if self.tts_service:
                    self.tts_service.set_speed(1.0)
                return {"action": "speed_change", "speed": 1.0, "success": True}

            elif any(phrase in command for phrase in ["new chat", "clear chat", "start over"]):
                return {"action": "new_chat", "success": True}

            elif any(phrase in command for phrase in ["summarize", "summary"]):
                return {"action": "summarize", "success": True}

            elif any(phrase in command for phrase in ["cite sources", "show sources"]):
                return {"action": "cite_sources", "success": True}

            elif any(phrase in command for phrase in ["delete last", "undo"]):
                return {"action": "delete_last", "success": True}

            else:
                self.logger.warning(f"Unknown voice command: '{command}'")
                return {"action": "unknown", "command": command, "success": False}

        except Exception as e:
            self.logger.error(f"Failed to process voice command: {str(e)}")
            return {"action": "error", "error": str(e), "success": False}

    def get_voice_status(self) -> Dict:
        """
        Get overall voice system status.

        Returns:
            Dict: Voice system status
        """
        return {
            "available": self.is_voice_available(),
            "stt_available": self.stt_service is not None,
            "tts_available": self.tts_service is not None,
            "active_sessions": len(self.voice_sessions),
            "metrics": self.get_voice_metrics(),
        }
