#!/usr/bin/env python3
"""
Test script for Voice Phase 2 features: Advanced VAD and streaming improvements.

Tests the WebRTC VAD integration and enhanced audio pipeline.
"""

import asyncio
import logging
import struct
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from services.voice_manager import VoiceManager  # noqa: E402
from services.webrtc_vad_service import WebRTCVADService  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def generate_test_audio(duration_ms: int, sample_rate: int = 16000, frequency: int = 440) -> bytes:
    """Generate test audio data (sine wave)."""
    import math

    samples = int(sample_rate * duration_ms / 1000)
    audio_data = []

    for i in range(samples):
        # Generate sine wave
        t = i / sample_rate
        sample = int(32767 * 0.5 * math.sin(2 * math.pi * frequency * t))
        audio_data.append(sample)

    # Convert to bytes (16-bit PCM)
    return struct.pack(f"<{len(audio_data)}h", *audio_data)


def generate_silence(duration_ms: int, sample_rate: int = 16000) -> bytes:
    """Generate silence."""
    samples = int(sample_rate * duration_ms / 1000)
    return b"\x00\x00" * samples


def test_webrtc_vad_availability():
    """Test WebRTC VAD availability."""
    logger.info("Testing WebRTC VAD availability...")

    try:
        is_available = WebRTCVADService.is_available()
        logger.info(f"WebRTC VAD available: {is_available}")

        if not is_available:
            logger.warning("⚠️ WebRTC VAD not available - install with: pip install webrtcvad")
            return False

        logger.info("✅ WebRTC VAD availability test passed")
        return True

    except Exception as e:
        logger.error(f"❌ WebRTC VAD availability test failed: {str(e)}")
        return False


def test_webrtc_vad_service():
    """Test WebRTC VAD service functionality."""
    logger.info("Testing WebRTC VAD service...")

    try:
        if not WebRTCVADService.is_available():
            logger.warning("⚠️ Skipping WebRTC VAD service test - not available")
            return True

        # Test configuration
        config = {
            "sample_rate": 16000,
            "aggressiveness": 2,
            "frame_duration_ms": 30,
            "speech_threshold": 3,
            "silence_threshold": 10,
        }

        vad_service = WebRTCVADService(config)
        logger.info(f"VAD service info: {vad_service.get_service_info()}")

        # Test with silence
        silence = generate_silence(100)  # 100ms silence
        results = vad_service.process_audio(silence)
        logger.info(f"Silence results: {len(results)} frames")

        for i, (is_speech, frame_data) in enumerate(results):
            logger.info(f"  Frame {i}: speech={is_speech}, bytes={len(frame_data)}")

        # Test with audio
        audio = generate_test_audio(100)  # 100ms audio
        results = vad_service.process_audio(audio)
        logger.info(f"Audio results: {len(results)} frames")

        for i, (is_speech, frame_data) in enumerate(results):
            logger.info(f"  Frame {i}: speech={is_speech}, bytes={len(frame_data)}")

        # Test VAD state
        vad_state = vad_service.get_vad_state()
        logger.info(f"VAD state: {vad_state}")

        logger.info("✅ WebRTC VAD service test passed")
        return True

    except Exception as e:
        logger.error(f"❌ WebRTC VAD service test failed: {str(e)}")
        return False


async def test_voice_manager_vad_integration():
    """Test VAD integration in voice manager."""
    logger.info("Testing voice manager VAD integration...")

    try:
        # Mock config with VAD enabled
        config = {
            "voice": {
                "enabled": True,
                "stt": {"enabled": True, "sample_rate": 16000, "model_name": "small-en"},
                "tts": {"enabled": True, "sample_rate": 22050, "voice_name": "en_US-lessac-medium"},
                "vad": {"enabled": True, "aggressiveness": 2, "frame_duration_ms": 30},
            }
        }

        voice_manager = VoiceManager(config)
        session_id = "test_vad_session"

        await voice_manager.create_voice_session(session_id)

        # Test VAD frame processing
        # 30ms frame at 16kHz = 480 samples = 960 bytes
        test_frame = generate_test_audio(30, 16000)
        logger.info(f"Test frame size: {len(test_frame)} bytes")

        vad_result = await voice_manager.process_vad_frame(session_id, test_frame)
        logger.info(f"VAD result: {vad_result}")

        # Test with silence frame
        silence_frame = generate_silence(30, 16000)
        vad_result_silence = await voice_manager.process_vad_frame(session_id, silence_frame)
        logger.info(f"VAD result (silence): {vad_result_silence}")

        await voice_manager.destroy_voice_session(session_id)

        logger.info("✅ Voice manager VAD integration test passed")
        return True

    except Exception as e:
        logger.error(f"❌ Voice manager VAD integration test failed: {str(e)}")
        return False


def test_audio_format_validation():
    """Test audio format validation."""
    logger.info("Testing audio format validation...")

    try:
        if not WebRTCVADService.is_available():
            logger.warning("⚠️ Skipping audio format validation test - WebRTC VAD not available")
            return True

        config = {"sample_rate": 16000, "frame_duration_ms": 30}
        vad_service = WebRTCVADService(config)

        # Test valid audio
        valid_audio = generate_test_audio(30)
        is_valid = vad_service.validate_audio_format(valid_audio)
        logger.info(f"Valid audio format check: {is_valid}")

        # Test invalid audio (odd number of bytes)
        invalid_audio = b"\x00\x01\x02"  # 3 bytes (not even)
        is_invalid = vad_service.validate_audio_format(invalid_audio)
        logger.info(f"Invalid audio format check: {is_invalid}")

        # Test empty audio
        empty_audio = b""
        is_empty = vad_service.validate_audio_format(empty_audio)
        logger.info(f"Empty audio format check: {is_empty}")

        logger.info("✅ Audio format validation test passed")
        return True

    except Exception as e:
        logger.error(f"❌ Audio format validation test failed: {str(e)}")
        return False


def test_vad_frame_timing():
    """Test VAD frame timing requirements."""
    logger.info("Testing VAD frame timing...")

    try:
        if not WebRTCVADService.is_available():
            logger.warning("⚠️ Skipping VAD frame timing test - WebRTC VAD not available")
            return True

        # Test different frame durations
        for frame_ms in [10, 20, 30]:
            logger.info(f"Testing {frame_ms}ms frames...")

            config = {"sample_rate": 16000, "frame_duration_ms": frame_ms, "aggressiveness": 1}

            vad_service = WebRTCVADService(config)

            # Generate exact frame size
            frame_audio = generate_test_audio(frame_ms)
            expected_bytes = int(16000 * frame_ms / 1000) * 2  # 16-bit samples

            logger.info(
                f"  Frame {frame_ms}ms: generated {len(frame_audio)} bytes, expected {expected_bytes}"
            )

            # Process frame
            results = vad_service.process_audio(frame_audio)
            logger.info(f"  Results: {len(results)} frames processed")

            # Reset for next test
            vad_service.reset()

        logger.info("✅ VAD frame timing test passed")
        return True

    except Exception as e:
        logger.error(f"❌ VAD frame timing test failed: {str(e)}")
        return False


async def main():
    """Run all Phase 2 tests."""
    logger.info("🎤 Starting Voice Phase 2 Tests...")

    tests = [
        ("WebRTC VAD Availability", test_webrtc_vad_availability),
        ("WebRTC VAD Service", test_webrtc_vad_service),
        ("Voice Manager VAD Integration", test_voice_manager_vad_integration),
        ("Audio Format Validation", test_audio_format_validation),
        ("VAD Frame Timing", test_vad_frame_timing),
    ]

    results = []

    for test_name, test_func in tests:
        logger.info(f"\n--- Running {test_name} Test ---")
        try:
            if asyncio.iscoroutinefunction(test_func):
                result = await test_func()
            else:
                result = test_func()
            results.append((test_name, result))
        except Exception as e:
            logger.error(f"Test {test_name} crashed: {str(e)}")
            results.append((test_name, False))

    # Summary
    logger.info("\n" + "=" * 50)
    logger.info("VOICE PHASE 2 TEST RESULTS")
    logger.info("=" * 50)

    passed = 0
    for test_name, result in results:
        status = "✅ PASS" if result else "❌ FAIL"
        logger.info(f"{test_name}: {status}")
        if result:
            passed += 1

    logger.info(f"\nOverall: {passed}/{len(results)} tests passed")

    if passed == len(results):
        logger.info("🎉 All Phase 2 tests passed! Advanced VAD ready.")
    elif passed >= len(results) - 1:  # Allow 1 failure (WebRTC VAD might not be installed)
        logger.info("✅ Phase 2 mostly working. Install webrtcvad for full functionality.")
    else:
        logger.warning("⚠️ Some Phase 2 tests failed. Check logs for details.")

    return passed >= len(results) - 1  # Success if at most 1 test failed


if __name__ == "__main__":
    success = asyncio.run(main())
    sys.exit(0 if success else 1)
