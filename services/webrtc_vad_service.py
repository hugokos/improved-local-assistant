"""
WebRTC VAD Service for improved voice activity detection.

This module provides enhanced VAD using webrtcvad library with proper
frame timing and aggressive/conservative modes.
"""

import logging
import struct
from typing import Optional, List, Tuple

try:
    import webrtcvad
    WEBRTC_VAD_AVAILABLE = True
except ImportError:
    webrtcvad = None
    WEBRTC_VAD_AVAILABLE = False


class WebRTCVADService:
    """
    Enhanced Voice Activity Detection using WebRTC VAD.
    
    Provides proper frame-based VAD with configurable aggressiveness
    and frame timing validation.
    """
    
    def __init__(self, config: dict = None):
        """
        Initialize WebRTC VAD service.
        
        Args:
            config: Configuration dictionary
        """
        self.config = config or {}
        self.logger = logging.getLogger(__name__)
        
        if not WEBRTC_VAD_AVAILABLE:
            raise ImportError("webrtcvad library not available. Install with: pip install webrtcvad")
        
        # VAD configuration - PROVEN SETTINGS from Home Assistant/Wyoming
        self.sample_rate = self.config.get("sample_rate", 16000)  # WebRTC VAD requires 16kHz
        self.aggressiveness = self.config.get("aggressiveness", 2)  # START AT 2, NOT 3 (too aggressive clips speech)
        self.frame_duration_ms = self.config.get("frame_duration_ms", 20)  # 20ms is sweet spot for most speech
        
        # Validate frame duration
        if self.frame_duration_ms not in [10, 20, 30]:
            raise ValueError(f"Frame duration must be 10, 20, or 30ms, got {self.frame_duration_ms}ms")
        
        # Calculate frame size in samples and bytes
        self.frame_samples = int(self.sample_rate * self.frame_duration_ms / 1000)
        self.frame_bytes = self.frame_samples * 2  # 16-bit PCM = 2 bytes per sample
        
        # Initialize VAD
        self.vad = webrtcvad.Vad(self.aggressiveness)
        
        # Frame buffer for accumulating partial frames
        self.frame_buffer = bytearray()
        
        # VAD state tracking
        self.speech_frames = 0
        self.silence_frames = 0
        self.is_speech_active = False
        
        # Smoothing parameters - HYSTERESIS for robust VAD
        self.speech_threshold = self.config.get("speech_threshold", 3)  # 3 voiced frames to start (60ms at 20ms frames)
        self.silence_threshold = self.config.get("silence_threshold", 15)  # 15 silence frames to end (300ms hangover)
        
        self.logger.info(f"🎙️ WebRTC VAD initialized: {self.sample_rate}Hz, {self.frame_duration_ms}ms frames, "
                        f"aggressiveness={self.aggressiveness} (PROVEN SETTINGS: start=3 frames, hangover=300ms)")
    
    def process_audio(self, audio_data: bytes) -> List[Tuple[bool, bytes]]:
        """
        Process audio data with strict WebRTC VAD frame validation.
        
        WebRTC VAD is VERY picky about frame timing and format:
        - Must be exactly 10/20/30ms frames
        - Must be 16-bit mono PCM at 8/16/32/48 kHz
        - Frame size = (sample_rate * frame_duration_ms / 1000) * 2 bytes
        
        Args:
            audio_data: Raw PCM16 audio data at 16kHz
            
        Returns:
            List of (is_speech, frame_data) tuples for complete frames
        """
        results = []
        
        # Validate input format first
        if not self.validate_audio_format(audio_data):
            self.logger.warning(f"Invalid audio format, skipping VAD processing")
            return results
        
        # Add new data to buffer
        self.frame_buffer.extend(audio_data)
        
        # Process complete frames with strict validation
        while len(self.frame_buffer) >= self.frame_bytes:
            # Extract one frame
            frame_data = bytes(self.frame_buffer[:self.frame_bytes])
            self.frame_buffer = self.frame_buffer[self.frame_bytes:]
            
            # CRITICAL: Validate exact frame size for WebRTC VAD
            if len(frame_data) != self.frame_bytes:
                self.logger.warning(f"❌ WebRTC VAD frame size mismatch: {len(frame_data)} bytes, expected {self.frame_bytes}")
                continue
            
            # Additional validation: check sample count
            sample_count = len(frame_data) // 2
            expected_samples = self.frame_samples
            if sample_count != expected_samples:
                self.logger.warning(f"❌ WebRTC VAD sample count mismatch: {sample_count} samples, expected {expected_samples}")
                continue
            
            # Run VAD on validated frame
            try:
                is_speech = self.vad.is_speech(frame_data, self.sample_rate)
                results.append((is_speech, frame_data))
                
                # Update smoothed VAD state with hysteresis
                self._update_vad_state(is_speech)
                
                # Debug logging for VAD decisions (occasionally)
                if len(results) % 50 == 0:  # Log every 50th frame (~1.5 seconds at 30ms frames)
                    self.logger.debug(f"🎙️ WebRTC VAD: speech={is_speech}, smoothed_active={self.is_speech_active}, "
                                    f"speech_frames={self.speech_frames}, silence_frames={self.silence_frames}")
                
            except Exception as e:
                self.logger.error(f"❌ WebRTC VAD processing error: {str(e)}")
                self.logger.error(f"Frame details: size={len(frame_data)}, expected={self.frame_bytes}, "
                                f"samples={len(frame_data)//2}, rate={self.sample_rate}")
                
                # Assume silence on error to be conservative
                results.append((False, frame_data))
                self._update_vad_state(False)
        
        return results
    
    def _update_vad_state(self, is_speech: bool):
        """Update smoothed VAD state with hysteresis."""
        if is_speech:
            self.speech_frames += 1
            self.silence_frames = 0
            
            # Transition to speech if we have enough consecutive speech frames
            if not self.is_speech_active and self.speech_frames >= self.speech_threshold:
                self.is_speech_active = True
                self.logger.debug(f"VAD: Speech started (after {self.speech_frames} frames)")
        else:
            self.silence_frames += 1
            self.speech_frames = 0
            
            # Transition to silence if we have enough consecutive silence frames
            if self.is_speech_active and self.silence_frames >= self.silence_threshold:
                self.is_speech_active = False
                self.logger.debug(f"VAD: Speech ended (after {self.silence_frames} frames)")
    
    def get_vad_state(self) -> dict:
        """
        Get current VAD state.
        
        Returns:
            Dict with VAD state information
        """
        return {
            "is_speech_active": self.is_speech_active,
            "speech_frames": self.speech_frames,
            "silence_frames": self.silence_frames,
            "frame_duration_ms": self.frame_duration_ms,
            "aggressiveness": self.aggressiveness,
            "buffer_bytes": len(self.frame_buffer)
        }
    
    def reset(self):
        """Reset VAD state."""
        self.frame_buffer.clear()
        self.speech_frames = 0
        self.silence_frames = 0
        self.is_speech_active = False
        self.logger.debug("VAD state reset")
    
    def validate_audio_format(self, audio_data: bytes) -> bool:
        """
        Validate that audio data is in the correct format for WebRTC VAD.
        
        Args:
            audio_data: Audio data to validate
            
        Returns:
            bool: True if format is valid
        """
        # Check that data length is even (16-bit samples)
        if len(audio_data) % 2 != 0:
            self.logger.warning(f"Audio data length not even: {len(audio_data)} bytes")
            return False
        
        # Check sample values are reasonable for 16-bit PCM
        try:
            samples = struct.unpack(f"<{len(audio_data)//2}h", audio_data)
            max_amplitude = max(abs(s) for s in samples) if samples else 0
            
            # Warn if audio is too quiet or too loud
            if max_amplitude < 100:
                self.logger.debug(f"Audio very quiet: max amplitude {max_amplitude}")
            elif max_amplitude > 30000:
                self.logger.debug(f"Audio very loud: max amplitude {max_amplitude}")
            
            return True
            
        except struct.error as e:
            self.logger.error(f"Audio format validation error: {str(e)}")
            return False
    
    @staticmethod
    def is_available() -> bool:
        """Check if WebRTC VAD is available."""
        return WEBRTC_VAD_AVAILABLE
    
    def get_service_info(self) -> dict:
        """Get service information."""
        return {
            "service": "WebRTCVADService",
            "available": WEBRTC_VAD_AVAILABLE,
            "sample_rate": self.sample_rate,
            "frame_duration_ms": self.frame_duration_ms,
            "frame_samples": self.frame_samples,
            "frame_bytes": self.frame_bytes,
            "aggressiveness": self.aggressiveness,
            "speech_threshold": self.speech_threshold,
            "silence_threshold": self.silence_threshold
        }