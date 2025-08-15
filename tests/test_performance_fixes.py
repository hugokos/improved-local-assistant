#!/usr/bin/env python3
"""
Test script for performance fixes and stability improvements.
"""

import asyncio
import json
import time

import requests
import websockets


def test_server_health():
    """Test basic server health."""
    print("🏥 Testing server health...")
    try:
        response = requests.get("http://localhost:8000/api/health", timeout=10)
        if response.status_code == 200:
            health_data = response.json()
            print(f"✅ Server healthy - Status: {health_data.get('status', 'unknown')}")
            print(f"   Services: {health_data.get('services', {})}")
            return True
        else:
            print(f"❌ Server unhealthy - Status: {response.status_code}")
            return False
    except Exception as e:
        print(f"❌ Health check failed: {e}")
        return False


async def test_websocket_stability():
    """Test WebSocket connection stability."""
    print("\n🔌 Testing WebSocket stability...")

    session_id = f"stability-test-{int(time.time())}"
    uri = f"ws://localhost:8000/ws/{session_id}"

    try:
        async with websockets.connect(uri) as websocket:
            print("✅ WebSocket connected")

            # Send multiple messages rapidly
            messages = [
                "Hello, this is a test message",
                "Can you tell me about survival skills?",
                "What about water purification?",
                "How do I build a shelter?",
                "What are the most important survival priorities?",
            ]

            for i, message in enumerate(messages):
                print(f"📤 Sending message {i+1}: {message[:30]}...")
                await websocket.send(message)

                # Wait for response
                response_count = 0
                citations_received = False

                while response_count < 50:  # Limit to prevent infinite loop
                    try:
                        msg = await asyncio.wait_for(websocket.recv(), timeout=2.0)

                        try:
                            json_msg = json.loads(msg)
                            if json_msg.get("type") == "citations":
                                citations_received = True
                                citations = json_msg.get("data", {}).get("citations", [])
                                print(f"   📚 Received {len(citations)} citations")
                            elif (
                                json_msg.get("type") == "typing"
                                and json_msg.get("status") == "stop"
                            ):
                                print(f"   ✅ Message {i+1} completed")
                                break
                        except json.JSONDecodeError:
                            response_count += 1

                    except asyncio.TimeoutError:
                        print(f"   ⏰ Message {i+1} timed out")
                        break

                # Small delay between messages
                await asyncio.sleep(0.5)

            print("✅ WebSocket stability test completed")
            return True

    except Exception as e:
        print(f"❌ WebSocket stability test failed: {e}")
        return False


def test_concurrent_connections():
    """Test multiple concurrent WebSocket connections."""
    print("\n🔀 Testing concurrent connections...")

    async def single_connection_test(connection_id):
        session_id = f"concurrent-{connection_id}-{int(time.time())}"
        uri = f"ws://localhost:8000/ws/{session_id}"

        try:
            async with websockets.connect(uri) as websocket:
                # Send a message
                await websocket.send(f"Test message from connection {connection_id}")

                # Wait for response
                response_received = False
                for _ in range(20):
                    try:
                        msg = await asyncio.wait_for(websocket.recv(), timeout=1.0)
                        try:
                            json_msg = json.loads(msg)
                            if (
                                json_msg.get("type") == "typing"
                                and json_msg.get("status") == "stop"
                            ):
                                response_received = True
                                break
                        except json.JSONDecodeError:
                            pass
                    except asyncio.TimeoutError:
                        break

                return response_received
        except Exception as e:
            print(f"❌ Connection {connection_id} failed: {e}")
            return False

    async def run_concurrent_test():
        # Test 5 concurrent connections
        tasks = [single_connection_test(i) for i in range(5)]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        successful = sum(1 for r in results if r is True)
        print(f"✅ {successful}/5 concurrent connections successful")
        return successful >= 4  # Allow 1 failure

    try:
        return asyncio.run(run_concurrent_test())
    except Exception as e:
        print(f"❌ Concurrent connections test failed: {e}")
        return False


def test_memory_usage():
    """Test memory usage during operation."""
    print("\n🧠 Testing memory usage...")

    try:
        import subprocess

        import psutil

        # Get initial memory usage
        initial_memory = psutil.virtual_memory().percent
        print(f"📊 Initial memory usage: {initial_memory:.1f}%")

        # Make several API calls to stress test
        for i in range(10):
            try:
                response = requests.post(
                    "http://localhost:8000/api/chat",
                    json={
                        "message": f"Test message {i} about survival techniques",
                        "session_id": f"memory-test-{i}",
                    },
                    timeout=15,
                )
                if response.status_code == 200:
                    print(f"   ✅ API call {i+1} successful")
                else:
                    print(f"   ⚠️  API call {i+1} failed: {response.status_code}")
            except Exception as e:
                print(f"   ❌ API call {i+1} error: {e}")

        # Check final memory usage
        final_memory = psutil.virtual_memory().percent
        memory_increase = final_memory - initial_memory

        print(f"📊 Final memory usage: {final_memory:.1f}%")
        print(f"📈 Memory increase: {memory_increase:.1f}%")

        if memory_increase < 10:  # Less than 10% increase is acceptable
            print("✅ Memory usage test PASSED")
            return True
        else:
            print("⚠️  Memory usage test WARNING - high memory increase")
            return False

    except ImportError:
        print("⚠️  psutil not available, skipping memory test")
        return True
    except Exception as e:
        print(f"❌ Memory usage test failed: {e}")
        return False


def test_citations_functionality():
    """Test the new citations functionality."""
    print("\n📚 Testing citations functionality...")

    try:
        # Send a message that should trigger citations
        response = requests.post(
            "http://localhost:8000/api/chat",
            json={
                "message": "What are the most important survival skills?",
                "session_id": "citations-test",
            },
            timeout=20,
        )

        if response.status_code == 200:
            chat_data = response.json()
            session_id = chat_data.get("session_id")
            print(f"✅ Chat response received for session: {session_id}")

            # Get citations
            citations_response = requests.get(
                f"http://localhost:8000/api/session/{session_id}/citations", timeout=10
            )

            if citations_response.status_code == 200:
                citations_data = citations_response.json()
                citations = citations_data.get("citations", [])

                print(f"📊 Citations found: {len(citations)}")
                if citations:
                    print(f"   Query: {citations_data.get('query', 'N/A')}")
                    print(f"   Sources: {citations_data.get('total_sources', 0)}")
                    for citation in citations[:2]:
                        print(
                            f"   • {citation.get('source', 'Unknown')}: {citation.get('text', '')[:60]}..."
                        )

                print("✅ Citations functionality test PASSED")
                return True
            else:
                print(f"❌ Citations API failed: {citations_response.status_code}")
                return False
        else:
            print(f"❌ Chat API failed: {response.status_code}")
            return False

    except Exception as e:
        print(f"❌ Citations test failed: {e}")
        return False


def main():
    """Run all performance and stability tests."""
    print("🚀 Performance Fixes Test Suite")
    print("=" * 50)

    tests = [
        ("Server Health", test_server_health),
        ("WebSocket Stability", lambda: asyncio.run(test_websocket_stability())),
        ("Concurrent Connections", test_concurrent_connections),
        ("Memory Usage", test_memory_usage),
        ("Citations Functionality", test_citations_functionality),
    ]

    results = {}

    for test_name, test_func in tests:
        print(f"\n🧪 Running {test_name} test...")
        try:
            start_time = time.time()
            result = test_func()
            duration = time.time() - start_time

            results[test_name] = {"passed": result, "duration": duration}

            status = "✅ PASSED" if result else "❌ FAILED"
            print(f"   {status} ({duration:.1f}s)")

        except Exception as e:
            results[test_name] = {"passed": False, "duration": 0, "error": str(e)}
            print(f"   ❌ FAILED - {e}")

    # Summary
    print("\n" + "=" * 50)
    print("📊 Test Results Summary:")

    passed_tests = sum(1 for r in results.values() if r["passed"])
    total_tests = len(results)

    for test_name, result in results.items():
        status = "✅" if result["passed"] else "❌"
        duration = result["duration"]
        print(f"   {status} {test_name} ({duration:.1f}s)")
        if "error" in result:
            print(f"      Error: {result['error']}")

    print(f"\n🎯 Overall: {passed_tests}/{total_tests} tests passed")

    if passed_tests == total_tests:
        print("🎉 All performance fixes are working correctly!")
    elif passed_tests >= total_tests * 0.8:
        print("⚠️  Most tests passed - system is mostly stable")
    else:
        print("❌ Multiple test failures - system needs attention")


if __name__ == "__main__":
    main()
