#!/usr/bin/env python3
"""
Test script to verify voice control fixes are working.

This script tests the binary audio frame handling between client and server.
"""

import array
import asyncio
import logging
import math
import struct
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from improved_local_assistant.services.vosk_stt_service import VoskSTTService  # noqa: E402

# Configure logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)


def rms16le(b: bytes) -> float:
    """Calculate RMS of 16-bit little-endian PCM audio."""
    if not b:
        return 0.0
    a = array.array("h")  # 16-bit signed
    a.frombytes(b)
    return math.sqrt(sum(x * x for x in a) / len(a))


def create_test_audio_frame(frequency=440, duration_ms=20, sample_rate=16000):
    """Create a test audio frame with a sine wave."""
    samples = int(sample_rate * duration_ms / 1000)  # 320 samples for 20ms at 16kHz
    frame = []

    for i in range(samples):
        t = i / sample_rate
        # Generate sine wave
        sample = int(32767 * 0.1 * math.sin(2 * math.pi * frequency * t))
        frame.append(sample)

    # Convert to bytes (little-endian 16-bit)
    audio_bytes = struct.pack("<" + "h" * len(frame), *frame)
    return audio_bytes


async def test_vosk_service():
    """Test the Vosk STT service with synthetic audio."""
    logger.info("Testing Vosk STT service...")

    try:
        # Initialize service
        stt_service = VoskSTTService()

        # Create recognizer for test session
        session_id = "test_session"
        success = await stt_service.create_recognizer(session_id)

        if not success:
            logger.error("Failed to create recognizer")
            return False

        logger.info("✅ Recognizer created successfully")

        # Test with synthetic audio frames
        logger.info("Testing with synthetic audio frames...")

        for i in range(10):
            # Create 20ms audio frame (640 bytes)
            audio_frame = create_test_audio_frame(frequency=440 + i * 50)

            # Verify frame size
            if len(audio_frame) != 640:
                logger.error(f"❌ Wrong frame size: {len(audio_frame)} bytes (expected 640)")
                return False

            # Calculate RMS
            rms = rms16le(audio_frame)
            logger.debug(f"Frame {i}: {len(audio_frame)} bytes, RMS: {rms:.1f}")

            # Process with STT
            result = await stt_service.process_audio(session_id, audio_frame)

            if result.get("error"):
                logger.error(f"❌ STT processing error: {result['error']}")
                return False

            if result.get("partial"):
                logger.info(f"📝 Partial result: '{result['partial']}'")

            if result.get("final"):
                logger.info(f"✅ Final result: '{result['final']}'")

        # Test silence frames
        logger.info("Testing with silence frames...")
        silence_frame = b"\x00" * 640  # 640 bytes of silence

        for i in range(5):
            rms = rms16le(silence_frame)
            logger.debug(f"Silence frame {i}: {len(silence_frame)} bytes, RMS: {rms:.1f}")

            result = await stt_service.process_audio(session_id, silence_frame)

            if result.get("error"):
                logger.error(f"❌ STT processing error: {result['error']}")
                return False

        # Cleanup
        await stt_service.destroy_recognizer(session_id)
        logger.info("✅ Test completed successfully")
        return True

    except Exception as e:
        logger.error(f"❌ Test failed: {e}")
        return False


def test_binary_frame_handling():
    """Test binary frame handling functions."""
    logger.info("Testing binary frame handling...")

    # Test RMS calculation
    test_frame = create_test_audio_frame()
    rms = rms16le(test_frame)

    logger.info(f"Test frame: {len(test_frame)} bytes, RMS: {rms:.1f}")

    if len(test_frame) != 640:
        logger.error(f"❌ Wrong frame size: {len(test_frame)} bytes")
        return False

    if rms < 1000:  # Should have significant amplitude
        logger.error(f"❌ RMS too low: {rms:.1f}")
        return False

    logger.info("✅ Binary frame handling test passed")
    return True


async def main():
    """Run all tests."""
    logger.info("🧪 Starting voice control fix tests...")

    # Test binary frame handling
    if not test_binary_frame_handling():
        logger.error("❌ Binary frame handling test failed")
        return 1

    # Test Vosk service
    if not await test_vosk_service():
        logger.error("❌ Vosk service test failed")
        return 1

    logger.info("🎉 All tests passed! Voice control fixes should be working.")
    return 0


if __name__ == "__main__":
    exit_code = asyncio.run(main())
    sys.exit(exit_code)
