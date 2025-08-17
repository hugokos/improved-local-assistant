#!/usr/bin/env python3
"""Basic usage examples for the Improved Local AI Assistant.

This module demonstrates common usage patterns and API interactions.
"""

import asyncio
import json

import websockets

from improved_local_assistant.core.http import http_session


def basic_rest_api_example() -> None:
    """Demonstrate basic REST API usage."""
    base_url = "http://localhost:8000"
    session = http_session()

    # Health check
    response = session.get(f"{base_url}/api/health")
    print(f"Health check: {response.json()}")

    # Send a chat message
    chat_data = {
        "message": "What is artificial intelligence?",
        "session_id": "example-session",
        "use_kg": True,
    }

    response = session.post(f"{base_url}/api/chat", json=chat_data)
    result = response.json()

    print(f"AI Response: {result['response']}")
    if result.get("sources"):
        print(f"Sources: {result['sources']}")


async def websocket_streaming_example() -> None:
    """Demonstrate WebSocket streaming for real-time responses."""
    uri = "ws://localhost:8000/ws/chat"

    async with websockets.connect(uri) as websocket:
        # Send a message
        message = {
            "message": "Explain machine learning in simple terms",
            "session_id": "websocket-example",
        }

        await websocket.send(json.dumps(message))

        # Receive streaming response
        full_response = ""
        async for response in websocket:
            data = json.loads(response)

            if data["type"] == "token":
                full_response += data["content"]
                print(data["content"], end="", flush=True)
            elif data["type"] == "complete":
                print("\n\nComplete response received.")
                if data.get("sources"):
                    print(f"Sources: {data['sources']}")
                break


def knowledge_graph_stats_example() -> None:
    """Demonstrate knowledge graph statistics retrieval."""
    session = http_session()
    response = session.get("http://localhost:8000/api/graph/stats")
    stats = response.json()

    print("Knowledge Graph Statistics:")
    print(f"  Entities: {stats.get('entities', 0)}")
    print(f"  Relationships: {stats.get('relationships', 0)}")
    print(f"  Last updated: {stats.get('last_updated', 'Never')}")


if __name__ == "__main__":
    print("=== Basic REST API Example ===")
    basic_rest_api_example()

    print("\n=== Knowledge Graph Stats ===")
    knowledge_graph_stats_example()

    print("\n=== WebSocket Streaming Example ===")
    asyncio.run(websocket_streaming_example())
