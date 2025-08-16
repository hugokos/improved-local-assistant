"""
Test script for the FastAPI application with WebSocket support.

This script provides command-line tools for testing the API endpoints,
WebSocket connections, and system monitoring features.
"""

import argparse
import asyncio
import json
import logging
import os
import sys
import time
from pprint import pprint

import requests
import websockets

# Fix for Windows asyncio event loop issue
if sys.platform == "win32":
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Default API URL
DEFAULT_API_URL = "http://localhost:8000"
DEFAULT_WS_URL = "ws://localhost:8000"


class APITester:
    """Test the FastAPI application endpoints."""

    def __init__(self, api_url: str = DEFAULT_API_URL):
        """Initialize the API tester."""
        self.api_url = api_url
        self.session = requests.Session()
        self.session_id = None

    def start_server(self):
        """Start the API server for testing."""
        try:
            # Check if server is already running
            try:
                response = self.session.get(f"{self.api_url}/health")
                if response.status_code == 200:
                    logger.info("Server is already running")
                    return True
            except (ConnectionError, requests.RequestException):
                pass

            # Start server in a separate process
            import subprocess
            import sys

            logger.info("Starting API server...")

            # Use Python executable from current environment
            python_exe = sys.executable

            # Start the server
            subprocess.Popen(
                [python_exe, "-m", "app.main"],
                cwd=os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            )

            # Wait for server to start
            max_retries = 10
            for i in range(max_retries):
                try:
                    response = self.session.get(f"{self.api_url}/health")
                    if response.status_code == 200:
                        logger.info("Server started successfully")
                        return True
                except (ConnectionError, requests.RequestException) as e:
                    logger.info(
                        f"Waiting for server to start ({i+1}/{max_retries})... Error: {str(e)}"
                    )
                    time.sleep(2)

            logger.error("Failed to start server")
            return False
        except Exception as e:
            logger.error(f"Error starting server: {str(e)}")
            return False

    def test_health(self):
        """Test the health check endpoint."""
        try:
            logger.info("Testing health check endpoint...")
            response = self.session.get(f"{self.api_url}/health")

            if response.status_code == 200:
                data = response.json()
                logger.info("Health check successful:")
                pprint(data)
                return True
            else:
                logger.error(f"Health check failed with status code {response.status_code}")
                return False
        except Exception as e:
            logger.error(f"Error testing health check: {str(e)}")
            return False

    def test_metrics(self):
        """Test the metrics endpoint."""
        try:
            logger.info("Testing metrics endpoint...")
            response = self.session.get(f"{self.api_url}/metrics")

            if response.status_code == 200:
                data = response.json()
                logger.info("Metrics retrieved successfully:")
                pprint(data)
                return True
            else:
                logger.error(f"Metrics retrieval failed with status code {response.status_code}")
                return False
        except Exception as e:
            logger.error(f"Error testing metrics: {str(e)}")
            return False

    def test_chat(self, message: str = "Hello, how are you?"):
        """Test the chat endpoint."""
        try:
            logger.info(f"Testing chat endpoint with message: '{message}'")

            payload = {"message": message, "session_id": self.session_id}

            response = self.session.post(f"{self.api_url}/api/chat", json=payload)

            if response.status_code == 200:
                data = response.json()
                logger.info("Chat response received:")
                print(f"Response: {data['response']}")

                # Store session ID for future requests
                self.session_id = data["session_id"]
                logger.info(f"Session ID: {self.session_id}")

                return True
            else:
                logger.error(f"Chat request failed with status code {response.status_code}")
                return False
        except Exception as e:
            logger.error(f"Error testing chat: {str(e)}")
            return False

    def test_graph_endpoints(self):
        """Test the knowledge graph endpoints."""
        try:
            logger.info("Testing knowledge graph query endpoint...")

            # Test graph query
            query_payload = {"query": "What is a knowledge graph?", "session_id": self.session_id}

            query_response = self.session.post(
                f"{self.api_url}/api/graph/query", json=query_payload
            )

            if query_response.status_code == 200:
                query_data = query_response.json()
                logger.info("Graph query successful:")
                print(f"Response: {query_data['response']}")
            else:
                logger.error(f"Graph query failed with status code {query_response.status_code}")
                return False

            # Test graph visualization
            logger.info("Testing knowledge graph visualization endpoint...")
            viz_response = self.session.get(f"{self.api_url}/api/graph/visualize")

            if viz_response.status_code == 200:
                logger.info("Graph visualization successful")

                # Save visualization to file for inspection
                with open("graph_visualization.html", "w") as f:
                    f.write(viz_response.text)

                logger.info("Visualization saved to graph_visualization.html")
            else:
                logger.error(
                    f"Graph visualization failed with status code {viz_response.status_code}"
                )
                return False

            return True
        except Exception as e:
            logger.error(f"Error testing graph endpoints: {str(e)}")
            return False

    def test_sessions(self):
        """Test the session management endpoints."""
        try:
            logger.info("Testing session management endpoints...")

            # List sessions
            list_response = self.session.get(f"{self.api_url}/api/sessions")

            if list_response.status_code == 200:
                list_data = list_response.json()
                logger.info("Sessions listed successfully:")
                pprint(list_data)
            else:
                logger.error(f"Session listing failed with status code {list_response.status_code}")
                return False

            # If we have a session ID, test getting session info
            if self.session_id:
                logger.info(f"Testing get session info for {self.session_id}...")
                info_response = self.session.get(f"{self.api_url}/api/session/{self.session_id}")

                if info_response.status_code == 200:
                    info_data = info_response.json()
                    logger.info("Session info retrieved successfully:")
                    pprint(info_data)
                else:
                    logger.error(
                        f"Session info retrieval failed with status code {info_response.status_code}"
                    )
                    return False

            return True
        except Exception as e:
            logger.error(f"Error testing sessions: {str(e)}")
            return False


class WebSocketTester:
    """Test WebSocket connections and streaming."""

    def __init__(self, ws_url: str = DEFAULT_WS_URL):
        """Initialize the WebSocket tester."""
        self.ws_url = ws_url
        self.session_id = f"test_{int(time.time())}"

    async def test_websocket(self):
        """Test WebSocket connection and message streaming."""
        try:
            logger.info(f"Testing WebSocket connection to {self.ws_url}/ws/{self.session_id}")

            async with websockets.connect(f"{self.ws_url}/ws/{self.session_id}") as websocket:
                logger.info("WebSocket connected successfully")

                # Send a test message
                test_message = "Hello, can you tell me about knowledge graphs?"
                logger.info(f"Sending message: '{test_message}'")
                await websocket.send(test_message)

                # Receive streaming response
                logger.info("Receiving streaming response:")
                response = ""

                while True:
                    try:
                        message = await asyncio.wait_for(websocket.recv(), timeout=1.0)

                        # Try to parse as JSON
                        try:
                            data = json.loads(message)
                            if "type" in data and data["type"] == "knowledge_graph":
                                logger.info("Received knowledge graph data")

                                # Save visualization to file for inspection
                                with open("websocket_graph.html", "w") as f:
                                    f.write(data["data"])

                                logger.info(
                                    "WebSocket graph visualization saved to websocket_graph.html"
                                )
                                break
                        except (json.JSONDecodeError, KeyError):
                            # Not JSON, treat as streaming text
                            print(message, end="", flush=True)
                            response += message
                    except asyncio.TimeoutError:
                        # No more messages
                        break

                print()  # New line after streaming response
                logger.info("WebSocket test completed successfully")
                return True
        except Exception as e:
            logger.error(f"Error testing WebSocket: {str(e)}")
            return False

    async def test_concurrent_users(self, num_users: int = 5):
        """Test WebSocket connections with multiple concurrent users."""
        try:
            logger.info(f"Testing {num_users} concurrent WebSocket connections")

            async def user_session(user_id):
                """Simulate a user session."""
                session_id = f"test_user_{user_id}_{int(time.time())}"

                try:
                    async with websockets.connect(f"{self.ws_url}/ws/{session_id}") as websocket:
                        logger.info(f"User {user_id} connected with session {session_id}")

                        # Send a test message
                        test_message = f"Hello, I am user {user_id}. What can you tell me?"
                        await websocket.send(test_message)

                        # Receive streaming response
                        response = ""

                        while True:
                            try:
                                message = await asyncio.wait_for(websocket.recv(), timeout=1.0)

                                # Try to parse as JSON
                                try:
                                    json.loads(message)
                                    # If JSON, we've received the knowledge graph data
                                    break
                                except (json.JSONDecodeError, KeyError):
                                    # Not JSON, treat as streaming text
                                    response += message
                            except asyncio.TimeoutError:
                                # No more messages
                                break

                        logger.info(f"User {user_id} received response of length {len(response)}")
                        return True
                except Exception as e:
                    logger.error(f"Error in user {user_id} session: {str(e)}")
                    return False

            # Create tasks for all users
            tasks = [user_session(i) for i in range(num_users)]

            # Run all tasks concurrently
            results = await asyncio.gather(*tasks)

            # Check if all tasks succeeded
            if all(results):
                logger.info("All concurrent user tests completed successfully")
                return True
            else:
                logger.error("Some concurrent user tests failed")
                return False
        except Exception as e:
            logger.error(f"Error testing concurrent users: {str(e)}")
            return False


async def main():
    """Main function for the API tester."""
    parser = argparse.ArgumentParser(description="Test the FastAPI application")
    parser.add_argument("--api-url", default=DEFAULT_API_URL, help="API URL")
    parser.add_argument("--ws-url", default=DEFAULT_WS_URL, help="WebSocket URL")
    parser.add_argument("--start-server", action="store_true", help="Start the API server")
    parser.add_argument("--test-health", action="store_true", help="Test health check endpoint")
    parser.add_argument("--test-metrics", action="store_true", help="Test metrics endpoint")
    parser.add_argument("--test-chat", action="store_true", help="Test chat endpoint")
    parser.add_argument("--test-graph-endpoints", action="store_true", help="Test graph endpoints")
    parser.add_argument("--test-sessions", action="store_true", help="Test session management")
    parser.add_argument("--test-websocket", action="store_true", help="Test WebSocket connection")
    parser.add_argument(
        "--test-concurrent", action="store_true", help="Test concurrent WebSocket connections"
    )
    parser.add_argument(
        "--num-users", type=int, default=5, help="Number of concurrent users to test"
    )
    parser.add_argument("--message", default="Hello, how are you?", help="Test message for chat")

    args = parser.parse_args()

    # Create API tester
    api_tester = APITester(api_url=args.api_url)

    # Start server if requested
    if args.start_server and not api_tester.start_server():
        return

    # Test health check
    if args.test_health:
        api_tester.test_health()

    # Test metrics
    if args.test_metrics:
        api_tester.test_metrics()

    # Test chat
    if args.test_chat:
        api_tester.test_chat(message=args.message)

    # Test graph endpoints
    if args.test_graph_endpoints:
        api_tester.test_graph_endpoints()

    # Test session management
    if args.test_sessions:
        api_tester.test_sessions()

    # Create WebSocket tester
    ws_tester = WebSocketTester(ws_url=args.ws_url)

    # Test WebSocket
    if args.test_websocket:
        await ws_tester.test_websocket()

    # Test concurrent WebSocket connections
    if args.test_concurrent:
        await ws_tester.test_concurrent_users(num_users=args.num_users)


if __name__ == "__main__":
    asyncio.run(main())
