#!/usr/bin/env python3
"""
Debug script to test voice control step by step.
"""

import logging
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# Configure logging to see all debug messages
logging.basicConfig(
    level=logging.DEBUG, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)


def main():
    print("🔧 Voice Control Debug Instructions")
    print("=" * 50)
    print()
    print("1. The server now has AGGRESSIVE logging enabled")
    print("2. The client now has detailed debugging enabled")
    print()
    print("Expected logs when you speak:")
    print()
    print("CLIENT (Browser Console):")
    print("- '🎵 Recorder message: vad_frame'")
    print("- '🎵 handleAudioFrame called: {...}'")
    print("- '🎵 Processing audio frame: {...}'")
    print("- '📤 Sent audio frame: 640 bytes, RMS: X.XXX'")
    print()
    print("SERVER (Terminal):")
    print("- '🎵 Starting audio loop for session [id]'")
    print("- '🎵 Frame 1 for [id]: len=640 bytes, RMS=XXXX.X'")
    print("- '🎤 Speech frame 1: len=640, RMS=XXXX.X'")
    print("- '📝 Partial result: [your speech]'")
    print()
    print("If you DON'T see these logs:")
    print("❌ No '🎵 Recorder message' → Worklet not sending messages")
    print("❌ No '🎵 handleAudioFrame called' → Message handler not called")
    print("❌ No '📤 Sent audio frame' → Frame not sent to server")
    print("❌ No '🎵 Frame X for [id]' → Server not receiving frames")
    print()
    print("Now restart your server and test voice control!")
    print("=" * 50)


if __name__ == "__main__":
    main()
