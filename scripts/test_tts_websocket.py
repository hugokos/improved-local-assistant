#!/usr/bin/env python3
"""
Test TTS WebSocket directly to debug audio issues.
"""

import asyncio
import json
import logging
import sys
from pathlib import Path

import websockets

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))


async def test_tts_websocket():
    """Test TTS WebSocket directly."""
    print("🧪 Testing TTS WebSocket")
    print("=" * 40)

    try:
        # Connect to TTS WebSocket
        uri = "ws://localhost:8000/ws/tts/test_session_123"
        print(f"Connecting to: {uri}")

        async with websockets.connect(uri) as websocket:
            print("✅ Connected to TTS WebSocket")

            # Send TTS request
            test_text = "Hello, this is a test of the text to speech system."
            message = {"type": "synthesize", "text": test_text}

            print(f"Sending TTS request: {test_text}")
            await websocket.send(json.dumps(message))

            # Receive responses
            audio_chunks = 0
            total_bytes = 0
            messages_received = []

            async for message in websocket:
                if isinstance(message, bytes):
                    # Audio data
                    audio_chunks += 1
                    total_bytes += len(message)
                    if audio_chunks == 1:
                        print(f"🔊 First audio chunk received: {len(message)} bytes")
                    elif audio_chunks % 10 == 0:
                        print(f"🔊 Received {audio_chunks} chunks, {total_bytes} bytes total")
                else:
                    # JSON message
                    try:
                        msg_data = json.loads(message)
                        messages_received.append(msg_data)
                        print(f"📨 Message: {msg_data}")

                        if msg_data.get("type") == "tts_end":
                            print(f"✅ TTS completed: {audio_chunks} chunks, {total_bytes} bytes")
                            break
                    except json.JSONDecodeError:
                        print(f"❌ Invalid JSON: {message}")

            print("\n📊 Summary:")
            print(f"  Audio chunks: {audio_chunks}")
            print(f"  Total bytes: {total_bytes}")
            print(f"  Messages: {len(messages_received)}")

            if audio_chunks > 0:
                print("✅ TTS WebSocket working correctly!")
            else:
                print("❌ No audio chunks received")

    except Exception as e:
        print(f"❌ Error testing TTS WebSocket: {e}")
        import traceback

        traceback.print_exc()


def main():
    """Main function."""
    logging.basicConfig(
        level=logging.DEBUG, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )

    print("🔍 TTS WebSocket Test")
    print("=" * 50)
    print("Make sure the server is running on localhost:8000")
    print()

    asyncio.run(test_tts_websocket())


if __name__ == "__main__":
    main()
