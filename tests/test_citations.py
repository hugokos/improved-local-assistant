#!/usr/bin/env python3
"""
Test script for the new citations functionality.
"""

import asyncio
import json

import requests
import websockets


async def test_citations_websocket():
    """Test citations via WebSocket."""
    session_id = "test-citations-session"
    uri = f"ws://localhost:8000/ws/{session_id}"

    print("🧪 Testing Citations via WebSocket")
    print(f"Session: {session_id}")

    try:
        async with websockets.connect(uri) as websocket:
            print("✅ Connected to WebSocket")

            # Wait for initial message
            initial_message = await websocket.recv()
            print(f"📨 Initial: {initial_message}")

            # Send a query that should trigger knowledge graph lookup
            test_query = "What are some essential survival skills?"
            print(f"📤 Sending: {test_query}")
            await websocket.send(test_query)

            # Collect all messages
            citations_received = False
            response_tokens = []

            while True:
                try:
                    message = await asyncio.wait_for(websocket.recv(), timeout=30.0)

                    try:
                        json_msg = json.loads(message)
                        if json_msg.get("type") == "citations":
                            print("🎯 Citations received!")
                            citations_data = json_msg.get("data", {})
                            citations = citations_data.get("citations", [])

                            print(f"📊 Total citations: {len(citations)}")
                            print(f"🔍 Query: {citations_data.get('query', 'N/A')}")
                            print(
                                f"⏱️  Query time: {citations_data.get('metadata', {}).get('query_time', 0)*1000:.1f}ms"
                            )

                            for i, citation in enumerate(citations[:3]):  # Show first 3
                                print(f"\n📖 Citation {citation.get('id', i+1)}:")
                                print(f"   Source: {citation.get('source', 'Unknown')}")
                                print(f"   Text: {citation.get('text', '')[:100]}...")
                                if citation.get("score"):
                                    print(f"   Relevance: {citation.get('score')*100:.1f}%")

                            citations_received = True
                        elif json_msg.get("type") == "typing":
                            if json_msg.get("status") == "start":
                                print("⌨️  Assistant is typing...")
                        else:
                            print(f"📨 Other message: {json_msg.get('type', 'unknown')}")
                    except json.JSONDecodeError:
                        # Response token
                        response_tokens.append(message)
                        if len(response_tokens) == 1:
                            print("💬 Response started...")

                except asyncio.TimeoutError:
                    print("⏰ Timeout - ending test")
                    break

            if citations_received:
                print("✅ Citations test PASSED")
            else:
                print("❌ Citations test FAILED - no citations received")

            if response_tokens:
                full_response = "".join(response_tokens)
                print(f"💬 Full response ({len(response_tokens)} tokens): {full_response[:200]}...")

    except Exception as e:
        print(f"❌ Error: {e}")


def test_citations_rest_api():
    """Test citations via REST API."""
    print("\n🧪 Testing Citations via REST API")

    try:
        # First, send a chat message
        chat_response = requests.post(
            "http://localhost:8000/api/chat",
            json={
                "message": "Tell me about water purification methods",
                "session_id": "test-api-session",
            },
            timeout=30,
        )

        if chat_response.status_code == 200:
            chat_data = chat_response.json()
            session_id = chat_data.get("session_id")
            print(f"✅ Chat response received for session: {session_id}")

            # Now get citations
            citations_response = requests.get(
                f"http://localhost:8000/api/session/{session_id}/citations", timeout=10
            )

            if citations_response.status_code == 200:
                citations_data = citations_response.json()
                citations = citations_data.get("citations", [])

                print(f"📊 REST API Citations: {len(citations)} found")
                for citation in citations[:2]:  # Show first 2
                    print(
                        f"   • {citation.get('source', 'Unknown')}: {citation.get('text', '')[:80]}..."
                    )

                print("✅ REST API citations test PASSED")
            else:
                print(f"❌ Citations API failed: {citations_response.status_code}")
        else:
            print(f"❌ Chat API failed: {chat_response.status_code}")

    except Exception as e:
        print(f"❌ REST API Error: {e}")


def main():
    """Run all citation tests."""
    print("🚀 Citations Functionality Test")
    print("=" * 50)

    # Test if server is running
    try:
        health_response = requests.get("http://localhost:8000/api/health", timeout=5)
        if health_response.status_code != 200:
            print("❌ Server not healthy - start with: python app.py")
            return
    except Exception:
        print("❌ Server not running - start with: python app.py")
        return

    print("✅ Server is running")

    # Run WebSocket test
    asyncio.run(test_citations_websocket())

    # Run REST API test
    test_citations_rest_api()

    print("\n" + "=" * 50)
    print("🎉 Citations testing completed!")


if __name__ == "__main__":
    main()
