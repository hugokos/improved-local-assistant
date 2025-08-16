#!/usr/bin/env python3
"""
Test script for voice features in the Improved Local AI Assistant.

This script tests the voice processing pipeline including STT and TTS services.
"""

import asyncio
import logging
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# Setup logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


async def test_voice_services():
    """Test voice services initialization and basic functionality."""
    print("🎤 Testing Voice Services")
    print("=" * 40)

    try:
        # Test configuration loading
        print("\n1. Loading configuration...")
        from app.core import load_config

        config = load_config()
        voice_config = config.get("voice", {})

        if not voice_config.get("enabled", False):
            print("   ⚠️  Voice features disabled in config")
            return False

        print("   ✅ Voice configuration loaded")

        # Test VoiceManager initialization
        print("\n2. Initializing VoiceManager...")
        from services.voice_manager import VoiceManager

        voice_manager = VoiceManager(config)

        if not voice_manager.is_voice_available():
            print("   ❌ Voice services not available")
            print("   💡 Run: python scripts/download_voice_models.py --all")
            return False

        print("   ✅ VoiceManager initialized")

        # Test voice session creation
        print("\n3. Testing voice session management...")
        session_id = "test_session_123"

        success = await voice_manager.create_voice_session(session_id)
        if not success:
            print("   ❌ Failed to create voice session")
            return False

        print("   ✅ Voice session created")

        # Test voice session state
        session_state = voice_manager.get_voice_session_state(session_id)
        if not session_state:
            print("   ❌ Failed to get voice session state")
            return False

        print("   ✅ Voice session state retrieved")

        # Test STT service
        print("\n4. Testing STT service...")
        if voice_manager.stt_service:
            stt_info = voice_manager.stt_service.get_service_info()
            print(f"   ✅ STT Service: {stt_info['service']}")
            print(f"   📊 Model: {stt_info.get('model_path', 'N/A')}")
            print(f"   🎯 Sample Rate: {stt_info.get('sample_rate', 'N/A')} Hz")
        else:
            print("   ❌ STT service not available")
            return False

        # Test TTS service
        print("\n5. Testing TTS service...")
        if voice_manager.tts_service:
            tts_info = voice_manager.tts_service.get_service_info()
            print(f"   ✅ TTS Service: {tts_info['service']}")
            voice_info = tts_info.get("voice_info", {})
            print(f"   🎵 Voice: {voice_info.get('voice_path', 'N/A')}")
            print(f"   🎯 Sample Rate: {voice_info.get('sample_rate', 'N/A')} Hz")
        else:
            print("   ❌ TTS service not available")
            return False

        # Test TTS synthesis (small test)
        print("\n6. Testing TTS synthesis...")
        test_text = "Hello, this is a test of the text-to-speech system."

        try:
            chunk_count = 0
            async for _audio_chunk in voice_manager.synthesize_response(session_id, test_text):
                chunk_count += 1
                if chunk_count >= 5:  # Just test first few chunks
                    break

            if chunk_count > 0:
                print(f"   ✅ TTS synthesis working ({chunk_count} chunks generated)")
            else:
                print("   ❌ No audio chunks generated")
                return False

        except Exception as e:
            print(f"   ❌ TTS synthesis failed: {str(e)}")
            return False

        # Test voice metrics
        print("\n7. Testing voice metrics...")
        metrics = voice_manager.get_voice_metrics()
        print(f"   📊 STT Requests: {metrics.get('stt_requests', 0)}")
        print(f"   📊 TTS Requests: {metrics.get('tts_requests', 0)}")
        print(f"   📊 Active Sessions: {metrics.get('active_voice_sessions', 0)}")
        print("   ✅ Voice metrics retrieved")

        # Cleanup
        print("\n8. Cleaning up...")
        await voice_manager.destroy_voice_session(session_id)
        print("   ✅ Voice session destroyed")

        print("\n🎉 All voice service tests passed!")
        return True

    except ImportError as e:
        print(f"   ❌ Import error: {str(e)}")
        print("   💡 Install voice dependencies: pip install vosk piper-tts sounddevice webrtcvad")
        return False
    except Exception as e:
        print(f"   ❌ Unexpected error: {str(e)}")
        import traceback

        traceback.print_exc()
        return False


def test_voice_models():
    """Test if voice models are available."""
    print("\n🎵 Testing Voice Models")
    print("=" * 40)

    project_root = Path(__file__).parent.parent

    # Check Vosk models
    print("\n1. Checking Vosk STT models...")
    vosk_dir = project_root / "models" / "vosk"

    if not vosk_dir.exists():
        print("   ❌ Vosk models directory not found")
        print("   💡 Run: python scripts/download_voice_models.py --vosk small-en")
        return False

    vosk_models = list(vosk_dir.glob("vosk-model-*"))
    if vosk_models:
        for model in vosk_models:
            print(f"   ✅ Found Vosk model: {model.name}")
    else:
        print("   ❌ No Vosk models found")
        return False

    # Check Piper voices
    print("\n2. Checking Piper TTS voices...")
    piper_dir = project_root / "models" / "piper"

    if not piper_dir.exists():
        print("   ❌ Piper voices directory not found")
        print("   💡 Run: python scripts/download_voice_models.py --piper en_US-lessac-medium")
        return False

    piper_voices = list(piper_dir.glob("*/*.onnx"))
    if piper_voices:
        for voice in piper_voices:
            print(f"   ✅ Found Piper voice: {voice.parent.name}")
    else:
        print("   ❌ No Piper voices found")
        return False

    print("\n✅ All voice models available!")
    return True


def test_voice_dependencies():
    """Test if voice processing dependencies are installed."""
    print("\n📦 Testing Voice Dependencies")
    print("=" * 40)

    dependencies = {
        "vosk": "Offline speech recognition",
        "piper": "Text-to-speech synthesis",
        "sounddevice": "Audio input/output",
        "webrtcvad": "Voice activity detection",
    }

    all_available = True

    for dep, description in dependencies.items():
        try:
            __import__(dep)
            print(f"   ✅ {dep}: {description}")
        except ImportError:
            print(f"   ❌ {dep}: {description} - NOT INSTALLED")
            all_available = False

    if not all_available:
        print("\n💡 Install missing dependencies:")
        print("   pip install vosk piper-tts sounddevice webrtcvad")
        return False

    print("\n✅ All voice dependencies available!")
    return True


async def main():
    """Main test function."""
    print("🎤 Voice Features Test Suite")
    print("=" * 50)

    # Test dependencies first
    if not test_voice_dependencies():
        print("\n❌ Voice dependencies test failed")
        return 1

    # Test models
    if not test_voice_models():
        print("\n❌ Voice models test failed")
        return 1

    # Test services
    if not await test_voice_services():
        print("\n❌ Voice services test failed")
        return 1

    print("\n🎉 All voice tests passed!")
    print("\nVoice chat functionality is ready to use!")
    return 0


if __name__ == "__main__":
    try:
        result = asyncio.run(main())
        sys.exit(result)
    except KeyboardInterrupt:
        print("\n⏹️  Test interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Test failed with error: {str(e)}")
        sys.exit(1)
