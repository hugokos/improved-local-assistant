#!/usr/bin/env python3
"""
Test script to verify voice control browser integration is working.

This script starts the server and provides instructions for testing voice control.
"""

import logging
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def print_test_instructions():
    """Print instructions for testing voice control."""
    print("\n" + "=" * 60)
    print("🎤 VOICE CONTROL INTEGRATION TEST")
    print("=" * 60)
    print()
    print("1. Start the server:")
    print("   python run_app.py")
    print()
    print("2. Open browser to: http://localhost:8000")
    print()
    print("3. Test voice control:")
    print("   • Click the microphone button or press Shift+M")
    print("   • You should see 'Listening...' and the orb should appear")
    print("   • Speak into your microphone")
    print("   • Check browser console for these logs:")
    print()
    print("   ✅ Expected browser console logs:")
    print("   - 'STT WebSocket connected'")
    print("   - '🎙️ STT ready'")
    print("   - '📤 Sent audio frame: 640 bytes, RMS: X.XXX'")
    print("   - '📝 Partial result: [your speech]'")
    print("   - '✅ Final result: [your speech]'")
    print()
    print("4. Check server logs for:")
    print("   ✅ Expected server logs:")
    print("   - 'STT session [id] ready'")
    print("   - 'Audio RMS=XXXX.X; len=640' (for speech)")
    print("   - 'Audio RMS ~0 (likely silence); len=640' (for silence)")
    print("   - 'Partial STT result for [id]: [text]'")
    print()
    print("5. Troubleshooting:")
    print("   ❌ If you see 'Audio RMS ~0' constantly:")
    print("   - Check microphone permissions")
    print("   - Try speaking louder")
    print("   - Check browser microphone settings")
    print()
    print("   ❌ If no audio frames are sent:")
    print("   - Check browser console for errors")
    print("   - Ensure HTTPS or localhost (required for microphone)")
    print("   - Try refreshing the page")
    print()
    print("   ❌ If WebSocket connection fails:")
    print("   - Check server is running on port 8000")
    print("   - Check firewall settings")
    print("   - Try different browser")
    print()
    print("=" * 60)
    print("🔧 FIXES APPLIED:")
    print("=" * 60)
    print("• Fixed WebSocket binary frame handling (iter_bytes)")
    print("• Fixed WebSocketDisconnect import")
    print("• Added RMS logging for audio verification")
    print("• Fixed client ArrayBuffer handling")
    print("• Ensured proper 640-byte frame sizes")
    print("• Removed missing voice-debug.js reference")
    print("• Fixed monitoring message type handling")
    print("=" * 60)
    print()


def main():
    """Main test function."""
    print_test_instructions()

    # Check if voice models are available
    models_dir = project_root / "models" / "vosk"
    if not models_dir.exists():
        print("⚠️  WARNING: Vosk models not found!")
        print("   Run: python scripts/download_voice_models.py --vosk small-en")
        print()

    # Check if server dependencies are available
    try:
        import vosk

        print("✅ Vosk library available")
    except ImportError:
        print("❌ Vosk library not installed!")
        print("   Run: pip install vosk")
        print()

    try:
        import fastapi

        print("✅ FastAPI available")
    except ImportError:
        print("❌ FastAPI not installed!")
        print("   Run: pip install fastapi")
        print()

    print("\nReady to test! Start the server and follow the instructions above.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
