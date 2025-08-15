#!/usr/bin/env python3
"""
Simple WebSocket chat test script to verify the session fix.
"""

import asyncio
import json
import uuid

import websockets


async def test_websocket_chat():
    """Test WebSocket chat functionality."""
    session_id = f"test-{uuid.uuid4()}"
    uri = f"ws://localhost:8000/ws/{session_id}"

    print(f"Testing WebSocket chat with session: {session_id}")

    try:
        async with websockets.connect(uri) as websocket:
            print("✅ Connected to WebSocket")

            # Wait for initial system message
            initial_message = await websocket.recv()
            print(f"📨 Received: {initial_message}")

            # Send a test message
            test_message = "Hello, this is a test message!"
            print(f"📤 Sending: {test_message}")
            await websocket.send(test_message)

            # Receive response tokens
            response_tokens = []
            timeout_count = 0
            max_timeouts = 10

            while timeout_count < max_timeouts:
                try:
                    message = await asyncio.wait_for(websocket.recv(), timeout=2.0)

                    try:
                        # Try to parse as JSON
                        json_msg = json.loads(message)
                        if json_msg.get("type") == "typing":
                            if json_msg.get("status") == "start":
                                print("⌨️  Assistant is typing...")
                            elif json_msg.get("status") == "stop":
                                print("✅ Assistant finished typing")
                        elif json_msg.get("type") == "error":
                            print(f"❌ Error: {json_msg.get('message')}")
                            break
                        else:
                            print(f"📨 JSON message: {json_msg}")
                    except json.JSONDecodeError:
                        # It's a text token
                        response_tokens.append(message)
                        print(f"🔤 Token: {repr(message)}")

                except asyncio.TimeoutError:
                    timeout_count += 1
                    if response_tokens:
                        print(
                            f"⏰ Timeout {timeout_count}/{max_timeouts} (received {len(response_tokens)} tokens so far)"
                        )
                    else:
                        print(f"⏰ Timeout {timeout_count}/{max_timeouts} (no response yet)")

            if response_tokens:
                full_response = "".join(response_tokens)
                print(f"✅ Complete response: {full_response}")
                print(f"📊 Total tokens received: {len(response_tokens)}")
            else:
                print("❌ No response received")

    except Exception as e:
        print(f"❌ Error: {e}")


if __name__ == "__main__":
    print("🧪 WebSocket Chat Test")
    print("Make sure the application is running with: python app.py")
    print()

    asyncio.run(test_websocket_chat())
