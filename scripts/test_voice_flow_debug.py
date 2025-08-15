#!/usr/bin/env python3
"""
Debug script for voice flow issues.
"""

import asyncio
import logging
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))


async def test_voice_services():
    """Test voice services individually."""
    print("🧪 Testing Voice Services")
    print("=" * 40)

    try:
        # Test TTS service
        print("🔊 Testing TTS Service...")
        from services.piper_tts_service import PiperTTSService

        from app.core.config import load_config

        config = load_config()
        tts_service = PiperTTSService(config)

        # Test synthesis
        test_text = "Hello, this is a test of the text to speech system."
        print(f"Synthesizing: '{test_text}'")

        audio_chunks = []
        async for chunk in tts_service.synthesize_stream(test_text, "test_session"):
            audio_chunks.append(chunk)

        total_bytes = sum(len(chunk) for chunk in audio_chunks)
        print(f"✅ TTS generated {len(audio_chunks)} chunks, {total_bytes} bytes total")

        # Test STT service
        print("\n🎤 Testing STT Service...")
        from services.vosk_stt_service import VoskSTTService

        stt_service = VoskSTTService(config)
        # STT service initializes automatically in constructor
        print("✅ STT service initialized")

        # Test Voice Manager
        print("\n🎙️ Testing Voice Manager...")
        from services.voice_manager import VoiceManager

        voice_manager = VoiceManager(config)
        # Voice manager initializes automatically in constructor
        print("✅ Voice Manager initialized")

        # Test synthesis through voice manager
        session_id = "test_session"
        print(f"Testing synthesis through voice manager for session: {session_id}")

        chunks = []
        async for chunk in voice_manager.synthesize_response(session_id, test_text):
            chunks.append(chunk)

        total_bytes = sum(len(chunk) for chunk in chunks)
        print(f"✅ Voice Manager generated {len(chunks)} chunks, {total_bytes} bytes total")

        print("\n✅ All voice services working correctly!")

    except Exception as e:
        print(f"❌ Error testing voice services: {e}")
        import traceback

        traceback.print_exc()


async def check_voice_config():
    """Check voice configuration."""
    print("🔧 Checking Voice Configuration")
    print("=" * 40)

    try:
        from app.core.config import load_config

        config = load_config()
        voice_config = config.get("voice", {})

        print(f"Voice enabled: {voice_config.get('enabled', False)}")
        print(f"STT enabled: {voice_config.get('stt', {}).get('enabled', False)}")
        print(f"TTS enabled: {voice_config.get('tts', {}).get('enabled', False)}")
        print(f"VAD enabled: {voice_config.get('vad', {}).get('enabled', False)}")

        # Check model paths
        stt_config = voice_config.get("stt", {})
        tts_config = voice_config.get("tts", {})

        print(f"\nSTT model path: {stt_config.get('model_path', 'Not set')}")
        print(f"TTS voice path: {tts_config.get('voice_path', 'Not set')}")

        # Check if model files exist
        import os

        stt_path = stt_config.get("model_path", "")
        tts_path = tts_config.get("voice_path", "")

        if stt_path and os.path.exists(stt_path):
            print(f"✅ STT model exists: {stt_path}")
        else:
            print(f"❌ STT model missing: {stt_path}")

        if tts_path and os.path.exists(tts_path):
            print(f"✅ TTS model exists: {tts_path}")
        else:
            print(f"❌ TTS model missing: {tts_path}")

    except Exception as e:
        print(f"❌ Error checking voice config: {e}")


def main():
    """Main function."""
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )

    print("🔍 Voice Flow Debug Tool")
    print("=" * 50)

    asyncio.run(check_voice_config())
    print()
    asyncio.run(test_voice_services())


if __name__ == "__main__":
    main()
