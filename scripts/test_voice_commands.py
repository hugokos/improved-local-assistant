#!/usr/bin/env python3
"""
Test script for voice command detection.
"""

import asyncio
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from services.voice_manager import VoiceManager


async def test_voice_commands():
    """Test voice command processing."""
    print("🎤 Testing Voice Command Processing")
    print("=" * 40)

    voice_manager = VoiceManager({"voice": {"stt": {"enabled": True}, "tts": {"enabled": True}}})

    test_commands = [
        "start over",
        "faster",
        "start over faster",
        "slow down",
        "new chat",
        "repeat",
        "stop",
        "summarize",
    ]

    for command in test_commands:
        result = await voice_manager.process_voice_command("test_session", command)
        action = result.get("action", "unknown")
        success = result.get("success", False)

        status = "✅" if success else "❌"
        print(f"{status} '{command}' → {action}")

    print("\n🎯 Command detection should now handle compound phrases like 'start over faster'")


if __name__ == "__main__":
    asyncio.run(test_voice_commands())
