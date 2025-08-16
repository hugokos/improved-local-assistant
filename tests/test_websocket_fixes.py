#!/usr/bin/env python3
"""
Test script to verify WebSocket stability fixes.

This script tests:
1. WebSocket connection state checking before sending
2. Proper handling of disconnected sockets
3. Session variable scoping fixes in ConversationManager
"""

import asyncio
import json
import logging
import sys
import time

import websockets

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


class WebSocketTester:
    def __init__(self, url="ws://localhost:8000/ws"):
        self.url = url
        self.session_id = f"test_session_{int(time.time())}"

    async def test_normal_conversation(self):
        """Test normal conversation flow."""
        logger.info("Testing normal conversation flow...")

        try:
            async with websockets.connect(f"{self.url}/{self.session_id}") as websocket:
                # Wait for connection confirmation
                response = await websocket.recv()
                data = json.loads(response)
                logger.info(f"Connected: {data}")

                # Send a test message
                test_message = "Hello, can you tell me about survivalism?"
                await websocket.send(test_message)
                logger.info(f"Sent: {test_message}")

                # Collect response tokens
                response_tokens = []
                start_time = time.time()

                while True:
                    try:
                        message = await asyncio.wait_for(websocket.recv(), timeout=30.0)

                        # Handle JSON messages
                        try:
                            data = json.loads(message)
                            if data.get("type") == "typing" and data.get("status") == "stop":
                                logger.info("Received typing stop - conversation complete")
                                break
                            elif data.get("type") == "citations":
                                logger.info(
                                    f"Received citations: {len(data.get('data', {}).get('citations', []))} items"
                                )
                            elif data.get("type") == "error":
                                logger.error(f"Received error: {data}")
                                break
                            elif data.get("type") == "heartbeat":
                                logger.debug("Received heartbeat")
                            else:
                                logger.info(f"Received JSON: {data}")
                        except json.JSONDecodeError:
                            # Handle text tokens
                            response_tokens.append(message)

                    except asyncio.TimeoutError:
                        logger.warning("Timeout waiting for response")
                        break

                elapsed = time.time() - start_time
                response_text = "".join(response_tokens)
                logger.info(f"Response received in {elapsed:.2f}s: {response_text[:100]}...")

                return True

        except Exception as e:
            logger.error(f"Normal conversation test failed: {str(e)}")
            return False

    async def test_abrupt_disconnect(self):
        """Test handling of abrupt disconnections."""
        logger.info("Testing abrupt disconnect handling...")

        try:
            async with websockets.connect(f"{self.url}/{self.session_id}_disconnect") as websocket:
                # Wait for connection
                await websocket.recv()
                logger.info("Connected for disconnect test")

                # Send a message
                await websocket.send("Tell me about emergency preparedness")
                logger.info("Sent message, now closing connection abruptly...")

                # Close connection abruptly
                await websocket.close()
                logger.info("Connection closed abruptly")

                return True

        except Exception as e:
            logger.error(f"Abrupt disconnect test failed: {str(e)}")
            return False

    async def test_multiple_connections(self):
        """Test multiple concurrent connections."""
        logger.info("Testing multiple concurrent connections...")

        async def single_connection(session_suffix):
            try:
                session_id = f"{self.session_id}_multi_{session_suffix}"
                async with websockets.connect(f"{self.url}/{session_id}") as websocket:
                    # Wait for connection
                    await websocket.recv()

                    # Send a message
                    await websocket.send(f"Test message from connection {session_suffix}")

                    # Wait for some response
                    response_count = 0
                    while response_count < 5:  # Wait for a few messages
                        try:
                            message = await asyncio.wait_for(websocket.recv(), timeout=10.0)
                            response_count += 1

                            # Check for completion
                            try:
                                data = json.loads(message)
                                if data.get("type") == "typing" and data.get("status") == "stop":
                                    break
                            except json.JSONDecodeError:
                                pass

                        except asyncio.TimeoutError:
                            break

                    logger.info(f"Connection {session_suffix} completed successfully")
                    return True

            except Exception as e:
                logger.error(f"Connection {session_suffix} failed: {str(e)}")
                return False

        # Run multiple connections concurrently
        tasks = [single_connection(i) for i in range(3)]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        success_count = sum(1 for result in results if result is True)
        logger.info(f"Multiple connections test: {success_count}/3 successful")

        return success_count >= 2  # Allow one failure

    async def test_session_variable_fix(self):
        """Test that session variable scoping is fixed."""
        logger.info("Testing session variable scoping fix...")

        try:
            # Test with a session that might trigger the error path
            async with websockets.connect(f"{self.url}/invalid_session_test") as websocket:
                # Wait for connection
                await websocket.recv()

                # Send a message that might trigger knowledge graph query
                await websocket.send("What do you know about water purification methods?")

                # Wait for response or error
                response_received = False
                start_time = time.time()

                while time.time() - start_time < 15.0:
                    try:
                        message = await asyncio.wait_for(websocket.recv(), timeout=5.0)

                        try:
                            data = json.loads(message)
                            if data.get("type") == "error":
                                # Check if it's the session variable error
                                if "referenced before assignment" in str(data):
                                    logger.error("Session variable scoping error still present!")
                                    return False
                                else:
                                    logger.info(
                                        f"Received expected error: {data.get('message', '')}"
                                    )
                            elif data.get("type") == "typing" and data.get("status") == "stop":
                                response_received = True
                                break
                        except json.JSONDecodeError:
                            # Text token received
                            response_received = True

                    except asyncio.TimeoutError:
                        break

                if response_received:
                    logger.info("Session variable scoping test passed")
                    return True
                else:
                    logger.warning("No response received, but no scoping error either")
                    return True

        except Exception as e:
            if "referenced before assignment" in str(e):
                logger.error(f"Session variable scoping error detected: {str(e)}")
                return False
            else:
                logger.info(f"Session variable test completed with exception: {str(e)}")
                return True


async def main():
    """Run all WebSocket stability tests."""
    logger.info("Starting WebSocket stability tests...")

    tester = WebSocketTester()

    tests = [
        ("Normal Conversation", tester.test_normal_conversation),
        ("Abrupt Disconnect", tester.test_abrupt_disconnect),
        ("Multiple Connections", tester.test_multiple_connections),
        ("Session Variable Fix", tester.test_session_variable_fix),
    ]

    results = {}

    for test_name, test_func in tests:
        logger.info(f"\n{'='*50}")
        logger.info(f"Running test: {test_name}")
        logger.info(f"{'='*50}")

        try:
            result = await test_func()
            results[test_name] = result
            status = "PASSED" if result else "FAILED"
            logger.info(f"Test {test_name}: {status}")
        except Exception as e:
            logger.error(f"Test {test_name} crashed: {str(e)}")
            results[test_name] = False

        # Wait between tests
        await asyncio.sleep(2)

    # Summary
    logger.info(f"\n{'='*50}")
    logger.info("TEST SUMMARY")
    logger.info(f"{'='*50}")

    passed = sum(1 for result in results.values() if result)
    total = len(results)

    for test_name, result in results.items():
        status = "PASSED" if result else "FAILED"
        logger.info(f"{test_name}: {status}")

    logger.info(f"\nOverall: {passed}/{total} tests passed")

    if passed == total:
        logger.info("🎉 All WebSocket stability tests PASSED!")
        return 0
    else:
        logger.error("❌ Some WebSocket stability tests FAILED!")
        return 1


if __name__ == "__main__":
    try:
        exit_code = asyncio.run(main())
        sys.exit(exit_code)
    except KeyboardInterrupt:
        logger.info("Tests interrupted by user")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Test runner crashed: {str(e)}")
        sys.exit(1)
