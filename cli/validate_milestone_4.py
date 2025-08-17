#!/usr/bin/env python3
"""
Interactive validation CLI for Milestone 4: API Layer and System Monitoring.

This script provides an interactive command-line interface for:
- Comprehensive API testing interface
- Interactive WebSocket connection testing
- API endpoint validation and performance testing
- Real-time monitoring of API operations
"""

import argparse
import asyncio
import contextlib
import json
import logging
import os
import sys
import time
from datetime import datetime
from pathlib import Path
from pprint import pprint

import psutil
import websockets
import yaml

from improved_local_assistant.core.http import http_session

# Add project root to Python path for imports
project_root = Path(__file__).resolve().parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

# Fix for Windows asyncio event loop issue
if sys.platform == "win32":
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Default API URL
DEFAULT_API_URL = "http://localhost:8000"
DEFAULT_WS_URL = "ws://localhost:8000"


def load_config():
    """Load configuration from config.yaml."""
    config_path = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "config.yaml"
    )

    if os.path.exists(config_path):
        with open(config_path) as f:
            return yaml.safe_load(f)
    else:
        logger.warning(f"Config file not found at {config_path}, using default configuration")
        return {
            "models": {
                "conversation": {
                    "name": "hermes3:3b",
                    "context_window": 8000,
                    "temperature": 0.7,
                    "max_tokens": 2048,
                },
                "knowledge": {
                    "name": "tinyllama",
                    "context_window": 2048,
                    "temperature": 0.2,
                    "max_tokens": 1024,
                },
            },
            "conversation": {"max_history_length": 50, "summarize_threshold": 20},
            "knowledge_graphs": {
                "prebuilt_directory": "./data/prebuilt_graphs",
                "dynamic_storage": "./data/dynamic_graph",
                "max_triplets_per_chunk": 4,
            },
            "ollama": {"host": "http://localhost:11434", "timeout": 120},
            "api": {"host": "localhost", "port": 8000, "cors_origins": ["*"]},
        }


class ValidationCLI:
    """Interactive CLI for validating Milestone 4: API Layer and System Monitoring."""

    def __init__(self):
        """Initialize the validation CLI."""
        self.config = load_config()
        self.api_url = DEFAULT_API_URL
        self.ws_url = DEFAULT_WS_URL
        self.session = http_session()
        self.session_id = None
        self.debug_mode = False
        self.memory_tracking = False
        self.memory_samples = []
        self.response_times = []
        self.active_ws_connections = {}
        self.monitoring_ws = None
        self.monitoring_task = None
        self.is_monitoring = False

        # Test results
        self.test_results = {
            "api_health": False,
            "api_metrics": False,
            "api_system_info": False,
            "api_chat": False,
            "api_sessions": False,
            "api_graph_endpoints": False,
            "websocket_connection": False,
            "websocket_chat": False,
            "websocket_monitoring": False,
            "concurrent_connections": False,
            "error_handling": False,
            "performance": False,
        }

    async def initialize(self):
        """Initialize components."""
        try:
            # Check if API server is running
            try:
                response = self.session.get(f"{self.api_url}/health")
                if response.status_code == 200:
                    logger.info("API server is running")
                    return True
                else:
                    logger.error(f"API server returned status code {response.status_code}")
                    return False
            except Exception as e:
                logger.error(f"Error connecting to API server: {str(e)}")

                # Ask if user wants to start the server
                print("\nAPI server is not running. Would you like to start it? (y/n)")
                choice = input("> ").strip().lower()

                if choice == "y":
                    return await self.start_server()
                else:
                    logger.error("Cannot proceed without API server")
                    return False

        except Exception as e:
            logger.error(f"Error initializing components: {str(e)}")
            import traceback

            traceback.print_exc()
            return False

    async def start_server(self):
        """Start the API server."""
        try:
            logger.info("Starting API server...")

            # Use Python executable from current environment
            python_exe = sys.executable

            # Start the server in a separate process
            import subprocess

            subprocess.Popen(
                [python_exe, "-m", "app.main"],
                cwd=os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            )

            # Wait for server to start
            max_retries = 30  # Increased from 10 to 30
            for i in range(max_retries):
                try:
                    response = self.session.get(f"{self.api_url}/health")
                    if response.status_code == 200:
                        logger.info("Server started successfully")
                        return True
                except ConnectionError as e:
                    logger.info(
                        f"Waiting for server to start ({i+1}/{max_retries})... Error: {str(e)}"
                    )
                    time.sleep(5)  # Increased from 2 to 5 seconds

            logger.error("Failed to start server")
            return False
        except Exception as e:
            logger.error(f"Error starting server: {str(e)}")
            return False

    def print_header(self):
        """Print the validation CLI header."""
        print("\n" + "=" * 80)
        print("MILESTONE 4 VALIDATION: API LAYER AND SYSTEM MONITORING".center(80))
        print("=" * 80)
        print(
            "\nThis interactive CLI allows you to validate the API layer and system monitoring functionality."
        )
        print("You can test API endpoints, WebSocket connections, and monitor system resources.")
        print("\nType '/help' to see available commands.")
        print("=" * 80 + "\n")

    def print_help(self):
        """Print help information."""
        print("\nAvailable Commands:")
        print("  /help                - Show this help message")
        print("  /exit                - Exit the validation CLI")
        print("  /test                - Run automated validation tests")
        print("  /status              - Show test status and results")
        print("  /debug <on|off>      - Toggle debug mode")
        print("  /memory <on|off>     - Toggle memory tracking")
        print("  /health              - Test health check endpoint")
        print("  /metrics             - Test metrics endpoint")
        print("  /system              - Test system info endpoint")
        print("  /chat <message>      - Test chat endpoint with a message")
        print("  /sessions            - Test session management endpoints")
        print("  /graphs              - Test knowledge graph endpoints")
        print("  /ws                  - Test WebSocket connection")
        print("  /ws_chat <message>   - Test WebSocket chat with a message")
        print("  /monitor <on|off>    - Start/stop real-time system monitoring")
        print("  /concurrent <n>      - Test n concurrent WebSocket connections")
        print("  /errors              - Test error handling")
        print("  /performance <n>     - Test API performance with n requests")
        print("  /report              - Generate validation report")
        print("\nAny other input will be treated as a message to the assistant via WebSocket.")

    def print_status(self):
        """Print test status and results."""
        print("\nTest Status:")
        passed = sum(1 for result in self.test_results.values() if result)
        print(f"Passed: {passed}/{len(self.test_results)} tests")

        for test, result in self.test_results.items():
            status = "✓" if result else "✗"
            print(f"{status} {test}")

    async def test_health(self):
        """Test the health check endpoint."""
        try:
            logger.info("Testing health check endpoint...")
            response = self.session.get(f"{self.api_url}/health")

            if response.status_code == 200:
                data = response.json()
                logger.info("Health check successful:")
                pprint(data)

                # Check required fields
                if all(
                    key in data
                    for key in [
                        "status",
                        "timestamp",
                        "services",
                        "models",
                        "knowledge_graphs",
                        "system",
                    ]
                ):
                    self.test_results["api_health"] = True
                    print("✓ Health check endpoint test passed")
                else:
                    print("✗ Health check response missing required fields")

                return True
            else:
                logger.error(f"Health check failed with status code {response.status_code}")
                print(f"✗ Health check failed with status code {response.status_code}")
                return False
        except Exception as e:
            logger.error(f"Error testing health check: {str(e)}")
            print(f"✗ Error testing health check: {str(e)}")
            return False

    async def test_metrics(self):
        """Test the metrics endpoint."""
        try:
            logger.info("Testing metrics endpoint...")
            response = self.session.get(f"{self.api_url}/metrics")

            if response.status_code == 200:
                data = response.json()
                logger.info("Metrics retrieved successfully:")
                pprint(data)

                # Check required fields
                if all(
                    key in data for key in ["timestamp", "resource_usage", "performance_metrics"]
                ):
                    self.test_results["api_metrics"] = True
                    print("✓ Metrics endpoint test passed")
                else:
                    print("✗ Metrics response missing required fields")

                return True
            else:
                logger.error(f"Metrics retrieval failed with status code {response.status_code}")
                print(f"✗ Metrics retrieval failed with status code {response.status_code}")
                return False
        except Exception as e:
            logger.error(f"Error testing metrics: {str(e)}")
            print(f"✗ Error testing metrics: {str(e)}")
            return False

    async def test_system_info(self):
        """Test the system info endpoint."""
        try:
            logger.info("Testing system info endpoint...")
            response = self.session.get(f"{self.api_url}/api/system/info")

            if response.status_code == 200:
                data = response.json()
                logger.info("System info retrieved successfully:")
                pprint(data)

                # Check required fields
                if all(
                    key in data
                    for key in [
                        "system_info",
                        "resource_usage",
                        "performance_metrics",
                        "resource_limits",
                        "health",
                    ]
                ):
                    self.test_results["api_system_info"] = True
                    print("✓ System info endpoint test passed")
                else:
                    print("✗ System info response missing required fields")

                return True
            else:
                logger.error(
                    f"System info retrieval failed with status code {response.status_code}"
                )
                print(f"✗ System info retrieval failed with status code {response.status_code}")
                return False
        except Exception as e:
            logger.error(f"Error testing system info: {str(e)}")
            print(f"✗ Error testing system info: {str(e)}")
            return False

    async def test_chat(self, message="Hello, how are you?"):
        """Test the chat endpoint."""
        try:
            logger.info(f"Testing chat endpoint with message: '{message}'")

            payload = {"message": message, "session_id": self.session_id}

            start_time = time.time()
            response = self.session.post(f"{self.api_url}/api/chat", json=payload)
            elapsed = time.time() - start_time
            self.response_times.append(elapsed)

            if response.status_code == 200:
                data = response.json()
                logger.info("Chat response received:")
                print(f"Response: {data['response']}")

                # Store session ID for future requests
                self.session_id = data["session_id"]
                logger.info(f"Session ID: {self.session_id}")

                # Check required fields
                if all(key in data for key in ["response", "session_id"]):
                    self.test_results["api_chat"] = True
                    print("✓ Chat endpoint test passed")
                else:
                    print("✗ Chat response missing required fields")

                print(f"Response time: {elapsed:.2f} seconds")
                return True
            else:
                logger.error(f"Chat request failed with status code {response.status_code}")
                print(f"✗ Chat request failed with status code {response.status_code}")
                return False
        except Exception as e:
            logger.error(f"Error testing chat: {str(e)}")
            print(f"✗ Error testing chat: {str(e)}")
            return False

    async def test_sessions(self):
        """Test the session management endpoints."""
        try:
            logger.info("Testing session management endpoints...")

            # List sessions
            list_response = self.session.get(f"{self.api_url}/api/sessions")

            if list_response.status_code == 200:
                list_data = list_response.json()
                logger.info("Sessions listed successfully:")
                pprint(list_data)

                # Check required fields
                if all(key in list_data for key in ["sessions", "count"]):
                    print("✓ Sessions listing successful")
                else:
                    print("✗ Sessions listing response missing required fields")
                    return False
            else:
                logger.error(f"Session listing failed with status code {list_response.status_code}")
                print(f"✗ Session listing failed with status code {list_response.status_code}")
                return False

            # If we have a session ID, test getting session info
            if self.session_id:
                logger.info(f"Testing get session info for {self.session_id}...")
                info_response = self.session.get(f"{self.api_url}/api/session/{self.session_id}")

                if info_response.status_code == 200:
                    info_data = info_response.json()
                    logger.info("Session info retrieved successfully:")
                    pprint(info_data)

                    # Check required fields
                    if all(
                        key in info_data for key in ["session_id", "created_at", "message_count"]
                    ):
                        print("✓ Session info retrieval successful")
                    else:
                        print("✗ Session info response missing required fields")
                        return False
                else:
                    logger.error(
                        f"Session info retrieval failed with status code {info_response.status_code}"
                    )
                    print(
                        f"✗ Session info retrieval failed with status code {info_response.status_code}"
                    )
                    return False

            self.test_results["api_sessions"] = True
            print("✓ Session management endpoints test passed")
            return True
        except Exception as e:
            logger.error(f"Error testing sessions: {str(e)}")
            print(f"✗ Error testing sessions: {str(e)}")
            return False

    async def test_graph_endpoints(self):
        """Test the knowledge graph endpoints."""
        try:
            logger.info("Testing knowledge graph endpoints...")

            # List graphs
            list_response = self.session.get(f"{self.api_url}/api/graphs")

            if list_response.status_code == 200:
                list_data = list_response.json()
                logger.info("Graphs listed successfully:")
                pprint(list_data)

                # Check required fields
                if all(
                    key in list_data
                    for key in ["graphs", "total_graphs", "total_nodes", "total_edges"]
                ):
                    print("✓ Graphs listing successful")
                else:
                    print("✗ Graphs listing response missing required fields")
                    return False
            else:
                logger.error(f"Graphs listing failed with status code {list_response.status_code}")
                print(f"✗ Graphs listing failed with status code {list_response.status_code}")
                return False

            # Test graph query
            query_payload = {"query": "What is a knowledge graph?", "session_id": self.session_id}

            query_response = self.session.post(
                f"{self.api_url}/api/graph/query", json=query_payload
            )

            if query_response.status_code == 200:
                query_data = query_response.json()
                logger.info("Graph query successful:")
                print(f"Response: {query_data['response']}")

                # Check required fields
                if all(key in query_data for key in ["response", "metadata"]):
                    print("✓ Graph query successful")
                else:
                    print("✗ Graph query response missing required fields")
                    return False
            else:
                logger.error(f"Graph query failed with status code {query_response.status_code}")
                print(f"✗ Graph query failed with status code {query_response.status_code}")
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
                print("✓ Graph visualization successful")
            else:
                logger.error(
                    f"Graph visualization failed with status code {viz_response.status_code}"
                )
                print(f"✗ Graph visualization failed with status code {viz_response.status_code}")
                return False

            self.test_results["api_graph_endpoints"] = True
            print("✓ Knowledge graph endpoints test passed")
            return True
        except Exception as e:
            logger.error(f"Error testing graph endpoints: {str(e)}")
            print(f"✗ Error testing graph endpoints: {str(e)}")
            return False

    async def test_websocket_connection(self):
        """Test WebSocket connection."""
        try:
            logger.info(
                f"Testing WebSocket connection to {self.ws_url}/ws/{self.session_id or 'test'}"
            )

            session_id = self.session_id or f"test_{int(time.time())}"

            async with websockets.connect(f"{self.ws_url}/ws/{session_id}") as websocket:
                logger.info("WebSocket connected successfully")
                print("✓ WebSocket connection successful")

                # Receive initial message
                try:
                    message = await asyncio.wait_for(websocket.recv(), timeout=2.0)
                    print(f"Received: {message}")
                except asyncio.TimeoutError:
                    print("No initial message received (timeout)")

                self.test_results["websocket_connection"] = True
                return True
        except Exception as e:
            logger.error(f"Error testing WebSocket connection: {str(e)}")
            print(f"✗ Error testing WebSocket connection: {str(e)}")
            return False

    async def test_websocket_chat(self, message="Hello via WebSocket"):
        """Test WebSocket chat."""
        try:
            logger.info(f"Testing WebSocket chat with message: '{message}'")

            session_id = self.session_id or f"test_{int(time.time())}"

            async with websockets.connect(f"{self.ws_url}/ws/{session_id}") as websocket:
                logger.info("WebSocket connected successfully")

                # Receive initial message
                with contextlib.suppress(asyncio.TimeoutError):
                    await asyncio.wait_for(websocket.recv(), timeout=2.0)

                # Send a test message
                logger.info(f"Sending message: '{message}'")
                await websocket.send(message)

                # Receive streaming response
                logger.info("Receiving streaming response:")
                print("Response: ", end="", flush=True)
                response = ""

                start_time = time.time()
                while time.time() - start_time < 30:  # Wait up to 30 seconds for complete response
                    try:
                        chunk = await asyncio.wait_for(websocket.recv(), timeout=1.0)

                        # Try to parse as JSON
                        try:
                            data = json.loads(chunk)
                            if "type" in data:
                                if data["type"] == "knowledge_graph":
                                    logger.info("Received knowledge graph data")

                                    # Save visualization to file for inspection
                                    with open("websocket_graph.html", "w") as f:
                                        f.write(data["data"])

                                    logger.info(
                                        "WebSocket graph visualization saved to websocket_graph.html"
                                    )
                                    print("\n✓ Received knowledge graph visualization")
                                elif data["type"] == "typing":
                                    if data["status"] == "stop":
                                        # End of response
                                        break
                                elif data["type"] == "system_status":
                                    print("\n✓ Received system status update")
                                elif data["type"] == "error":
                                    print(f"\n✗ Error: {data['message']}")
                                    break
                        except (json.JSONDecodeError, KeyError):
                            # Not JSON, treat as streaming text
                            print(chunk, end="", flush=True)
                            response += chunk
                    except asyncio.TimeoutError:
                        # No more messages for now
                        if response:
                            break

                print()  # New line after streaming response

                if response:
                    self.test_results["websocket_chat"] = True
                    print("✓ WebSocket chat test passed")
                else:
                    print("✗ No text response received")

                return True
        except Exception as e:
            logger.error(f"Error testing WebSocket chat: {str(e)}")
            print(f"✗ Error testing WebSocket chat: {str(e)}")
            return False

    async def start_monitoring(self):
        """Start real-time system monitoring via WebSocket."""
        if self.is_monitoring:
            print("Monitoring is already active")
            return

        try:
            logger.info("Starting real-time system monitoring...")

            # Connect to monitoring WebSocket
            self.monitoring_ws = await websockets.connect(f"{self.ws_url}/ws/monitor")
            logger.info("Monitoring WebSocket connected successfully")

            # Start monitoring task
            self.is_monitoring = True
            self.monitoring_task = asyncio.create_task(self._monitoring_loop())

            print("✓ Real-time system monitoring started")
            self.test_results["websocket_monitoring"] = True
            return True
        except Exception as e:
            logger.error(f"Error starting monitoring: {str(e)}")
            print(f"✗ Error starting monitoring: {str(e)}")
            return False

    async def stop_monitoring(self):
        """Stop real-time system monitoring."""
        if not self.is_monitoring:
            print("Monitoring is not active")
            return

        try:
            logger.info("Stopping real-time system monitoring...")

            # Stop monitoring task
            self.is_monitoring = False
            if self.monitoring_task:
                self.monitoring_task.cancel()
                with contextlib.suppress(asyncio.CancelledError):
                    await self.monitoring_task
                self.monitoring_task = None

            # Close WebSocket connection
            if self.monitoring_ws:
                await self.monitoring_ws.close()
                self.monitoring_ws = None

            print("✓ Real-time system monitoring stopped")
            return True
        except Exception as e:
            logger.error(f"Error stopping monitoring: {str(e)}")
            print(f"✗ Error stopping monitoring: {str(e)}")
            return False

    async def _monitoring_loop(self):
        """Background task for receiving monitoring updates."""
        try:
            while self.is_monitoring and self.monitoring_ws:
                try:
                    # Receive message
                    message = await self.monitoring_ws.recv()

                    # Parse JSON
                    try:
                        data = json.loads(message)

                        if "type" in data and data["type"] == "system_status":
                            # Extract resource usage
                            if "resource_usage" in data:
                                cpu = data["resource_usage"].get("cpu_percent", 0)
                                memory = data["resource_usage"].get("memory_percent", 0)

                                # Print status
                                print(
                                    f"\rCPU: {cpu:5.1f}% | Memory: {memory:5.1f}% | {datetime.now().strftime('%H:%M:%S')}",
                                    end="",
                                    flush=True,
                                )
                    except (json.JSONDecodeError, KeyError, ValueError):
                        # Not JSON or invalid format
                        pass

                except asyncio.CancelledError:
                    break
                except Exception as e:
                    logger.error(f"Error in monitoring loop: {str(e)}")
                    await asyncio.sleep(1)
        except asyncio.CancelledError:
            pass
        except Exception as e:
            logger.error(f"Monitoring loop terminated with error: {str(e)}")

    async def test_concurrent_connections(self, num_connections=3):
        """Test multiple concurrent WebSocket connections."""
        try:
            logger.info(f"Testing {num_connections} concurrent WebSocket connections...")

            async def connect_and_chat(session_id):
                try:
                    async with websockets.connect(f"{self.ws_url}/ws/{session_id}") as websocket:
                        # Receive initial message
                        with contextlib.suppress(asyncio.TimeoutError):
                            await asyncio.wait_for(websocket.recv(), timeout=2.0)

                        # Send a message
                        await websocket.send(f"Hello from session {session_id}")

                        # Wait for response
                        response = ""
                        start_time = time.time()
                        while time.time() - start_time < 10:
                            try:
                                message = await asyncio.wait_for(websocket.recv(), timeout=1.0)

                                # Try to parse as JSON
                                try:
                                    json.loads(message)
                                except (json.JSONDecodeError, KeyError):
                                    # Not JSON, treat as streaming text
                                    response += message
                            except asyncio.TimeoutError:
                                if response:
                                    break

                        return len(response) > 0
                except Exception as e:
                    logger.error(f"Error in connection {session_id}: {str(e)}")
                    return False

            # Create tasks for multiple connections
            tasks = []
            for i in range(num_connections):
                session_id = f"test_concurrent_{i}_{int(time.time())}"
                tasks.append(connect_and_chat(session_id))

            # Run all tasks concurrently
            results = await asyncio.gather(*tasks)

            # Check results
            success_count = sum(1 for r in results if r)
            logger.info(f"{success_count}/{num_connections} concurrent connections successful")
            print(f"✓ {success_count}/{num_connections} concurrent connections successful")

            if success_count >= num_connections // 2:
                self.test_results["concurrent_connections"] = True
                print("✓ Concurrent connections test passed")
            else:
                print("✗ Too many connection failures")

            return success_count == num_connections
        except Exception as e:
            logger.error(f"Error testing concurrent connections: {str(e)}")
            print(f"✗ Error testing concurrent connections: {str(e)}")
            return False

    async def test_error_handling(self):
        """Test error handling in API endpoints."""
        try:
            logger.info("Testing error handling...")
            errors_found = 0

            # Test invalid session ID
            logger.info("Testing invalid session ID...")
            invalid_session = self.session.get(f"{self.api_url}/api/session/invalid_session_id")

            if invalid_session.status_code == 404:
                print("✓ Invalid session ID correctly returns 404")
                errors_found += 1
            else:
                print(f"✗ Invalid session ID returned {invalid_session.status_code}, expected 404")

            # Test invalid graph ID
            logger.info("Testing invalid graph ID...")
            invalid_graph = self.session.get(f"{self.api_url}/api/graph/invalid_graph_id/stats")

            if invalid_graph.status_code == 404:
                print("✓ Invalid graph ID correctly returns 404")
                errors_found += 1
            else:
                print(f"✗ Invalid graph ID returned {invalid_graph.status_code}, expected 404")

            # Test invalid JSON payload
            logger.info("Testing invalid JSON payload...")
            session = http_session()
            invalid_json = session.post(
                f"{self.api_url}/api/chat",
                data="invalid json",
                headers={"Content-Type": "application/json"},
            )

            if invalid_json.status_code == 422:  # Unprocessable Entity
                print("✓ Invalid JSON payload correctly returns 422")
                errors_found += 1
            else:
                print(f"✗ Invalid JSON payload returned {invalid_json.status_code}, expected 422")

            # Test missing required field
            logger.info("Testing missing required field...")
            missing_field = self.session.post(
                f"{self.api_url}/api/chat",
                json={"session_id": self.session_id},  # Missing 'message' field
            )

            if missing_field.status_code in [400, 422]:
                print("✓ Missing required field correctly returns error")
                errors_found += 1
            else:
                print(
                    f"✗ Missing required field returned {missing_field.status_code}, expected 400 or 422"
                )

            if errors_found >= 3:
                self.test_results["error_handling"] = True
                print("✓ Error handling test passed")
            else:
                print("✗ Error handling test failed")

            return errors_found >= 3
        except Exception as e:
            logger.error(f"Error testing error handling: {str(e)}")
            print(f"✗ Error testing error handling: {str(e)}")
            return False

    async def test_performance(self, num_requests=10):
        """Test API performance under load."""
        try:
            logger.info(f"Testing API performance with {num_requests} requests...")

            # Test endpoint to use
            endpoint = f"{self.api_url}/health"

            # Send requests sequentially and measure time
            start_time = time.time()
            responses = []

            for i in range(num_requests):
                try:
                    response = self.session.get(endpoint)
                    responses.append(response)
                    print(f"\rCompleted {i+1}/{num_requests} requests...", end="", flush=True)
                except Exception as e:
                    logger.error(f"Request {i+1} failed: {str(e)}")

            end_time = time.time()
            print()  # New line after progress

            # Calculate statistics
            total_time = end_time - start_time
            success_count = sum(1 for r in responses if r.status_code == 200)
            avg_time = total_time / num_requests if num_requests > 0 else 0

            logger.info(f"Performance test: {success_count}/{num_requests} successful requests")
            logger.info(f"Total time: {total_time:.2f} seconds")
            logger.info(f"Average request time: {avg_time:.4f} seconds")

            print(f"✓ {success_count}/{num_requests} successful requests")
            print(f"✓ Total time: {total_time:.2f} seconds")
            print(f"✓ Average request time: {avg_time:.4f} seconds")

            # All requests should succeed and average time should be reasonable
            if (
                success_count == num_requests and avg_time < 1.0
            ):  # Less than 1 second per request on average
                self.test_results["performance"] = True
                print("✓ Performance test passed")
            else:
                print("✗ Performance test failed")

            return success_count == num_requests
        except Exception as e:
            logger.error(f"Error testing performance: {str(e)}")
            print(f"✗ Error testing performance: {str(e)}")
            return False

    async def generate_report(self):
        """Generate a validation report."""
        try:
            logger.info("Generating validation report...")

            # Get system info
            try:
                system_response = self.session.get(f"{self.api_url}/api/system/info")
                system_info = system_response.json() if system_response.status_code == 200 else {}
            except (ConnectionError, json.JSONDecodeError) as e:
                logger.warning(f"Could not get system info: {str(e)}")
                system_info = {}

            # Calculate test results
            passed = sum(1 for result in self.test_results.values() if result)
            total = len(self.test_results)
            pass_rate = (passed / total) * 100 if total > 0 else 0

            # Calculate performance metrics
            avg_response_time = (
                sum(self.response_times) / len(self.response_times) if self.response_times else 0
            )

            # Generate report
            report = {
                "timestamp": datetime.now().isoformat(),
                "test_results": {
                    "passed": passed,
                    "total": total,
                    "pass_rate": pass_rate,
                    "tests": self.test_results,
                },
                "performance": {
                    "average_response_time": avg_response_time,
                    "response_times": self.response_times,
                },
                "system_info": system_info,
            }

            # Save report to file
            report_file = f"validation_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
            with open(report_file, "w") as f:
                json.dump(report, f, indent=2)

            logger.info(f"Report saved to {report_file}")
            print(f"✓ Validation report saved to {report_file}")

            # Print summary
            print("\nValidation Report Summary:")
            print(f"Tests Passed: {passed}/{total} ({pass_rate:.1f}%)")
            print(f"Average Response Time: {avg_response_time:.4f} seconds")

            return True
        except Exception as e:
            logger.error(f"Error generating report: {str(e)}")
            print(f"✗ Error generating report: {str(e)}")
            return False

    async def run_automated_tests(self):
        """Run automated validation tests."""
        print("\nRunning automated validation tests...")

        # Test 1: Health check
        print("\n[TEST 1/10] Health check endpoint...")
        await self.test_health()

        # Test 2: Metrics endpoint
        print("\n[TEST 2/10] Metrics endpoint...")
        await self.test_metrics()

        # Test 3: System info endpoint
        print("\n[TEST 3/10] System info endpoint...")
        await self.test_system_info()

        # Test 4: Chat endpoint
        print("\n[TEST 4/10] Chat endpoint...")
        await self.test_chat("Hello, I'm testing the API layer")

        # Test 5: Session management
        print("\n[TEST 5/10] Session management...")
        await self.test_sessions()

        # Test 6: Knowledge graph endpoints
        print("\n[TEST 6/10] Knowledge graph endpoints...")
        await self.test_graph_endpoints()

        # Test 7: WebSocket connection
        print("\n[TEST 7/10] WebSocket connection...")
        await self.test_websocket_connection()

        # Test 8: WebSocket chat
        print("\n[TEST 8/10] WebSocket chat...")
        await self.test_websocket_chat("Hello via WebSocket, testing streaming responses")

        # Test 9: Error handling
        print("\n[TEST 9/10] Error handling...")
        await self.test_error_handling()

        # Test 10: Performance
        print("\n[TEST 10/10] Performance testing...")
        await self.test_performance(5)  # 5 requests for quick testing

        # Print summary
        print("\nTest Summary:")
        self.print_status()

    async def handle_command(self, command, args):
        """Handle a command."""
        if command == "/help":
            self.print_help()
            return True

        elif command == "/exit":
            # Stop monitoring if active
            if self.is_monitoring:
                await self.stop_monitoring()
            return False

        elif command == "/test":
            await self.run_automated_tests()
            return True

        elif command == "/status":
            self.print_status()
            return True

        elif command == "/debug":
            if args and args[0].lower() in ["on", "true", "yes", "1"]:
                self.debug_mode = True
                print("Debug mode enabled")
            elif args and args[0].lower() in ["off", "false", "no", "0"]:
                self.debug_mode = False
                print("Debug mode disabled")
            else:
                self.debug_mode = not self.debug_mode
                print(f"Debug mode {'enabled' if self.debug_mode else 'disabled'}")
            return True

        elif command == "/memory":
            if args and args[0].lower() in ["on", "true", "yes", "1"]:
                self.memory_tracking = True
                print("Memory tracking enabled")
            elif args and args[0].lower() in ["off", "false", "no", "0"]:
                self.memory_tracking = False
                print("Memory tracking disabled")
            else:
                self.memory_tracking = not self.memory_tracking
                print(f"Memory tracking {'enabled' if self.memory_tracking else 'disabled'}")
            return True

        elif command == "/health":
            await self.test_health()
            return True

        elif command == "/metrics":
            await self.test_metrics()
            return True

        elif command == "/system":
            await self.test_system_info()
            return True

        elif command == "/chat":
            message = " ".join(args) if args else "Hello, how are you?"
            await self.test_chat(message)
            return True

        elif command == "/sessions":
            await self.test_sessions()
            return True

        elif command == "/graphs":
            await self.test_graph_endpoints()
            return True

        elif command == "/ws":
            await self.test_websocket_connection()
            return True

        elif command == "/ws_chat":
            message = " ".join(args) if args else "Hello via WebSocket"
            await self.test_websocket_chat(message)
            return True

        elif command == "/monitor":
            if args and args[0].lower() in ["on", "true", "yes", "1", "start"]:
                await self.start_monitoring()
            elif args and args[0].lower() in ["off", "false", "no", "0", "stop"]:
                await self.stop_monitoring()
            else:
                if self.is_monitoring:
                    await self.stop_monitoring()
                else:
                    await self.start_monitoring()
            return True

        elif command == "/concurrent":
            num = int(args[0]) if args and args[0].isdigit() else 3
            await self.test_concurrent_connections(num)
            return True

        elif command == "/errors":
            await self.test_error_handling()
            return True

        elif command == "/performance":
            num = int(args[0]) if args and args[0].isdigit() else 10
            await self.test_performance(num)
            return True

        elif command == "/report":
            await self.generate_report()
            return True

        return True

    async def process_message(self, message):
        """Process a user message via WebSocket."""
        try:
            # Create session ID if needed
            if not self.session_id:
                self.session_id = f"cli_{int(time.time())}"

            # Connect to WebSocket
            async with websockets.connect(f"{self.ws_url}/ws/{self.session_id}") as websocket:
                # Receive initial message if any
                try:
                    initial = await asyncio.wait_for(websocket.recv(), timeout=1.0)
                    if self.debug_mode:
                        print(f"[DEBUG] Initial message: {initial}")
                except asyncio.TimeoutError:
                    pass

                # Send message
                await websocket.send(message)

                # Receive streaming response
                print("Assistant: ", end="", flush=True)
                response = ""

                start_time = time.time()
                while True:
                    try:
                        chunk = await asyncio.wait_for(websocket.recv(), timeout=1.0)

                        # Try to parse as JSON
                        try:
                            data = json.loads(chunk)
                            if "type" in data:
                                if data["type"] == "knowledge_graph":
                                    print("\n[Knowledge Graph received]")
                                elif data["type"] == "system_status" and self.debug_mode:
                                    print(
                                        f"\n[DEBUG] System status: CPU {data['resource_usage']['cpu_percent']}%, Memory {data['resource_usage']['memory_percent']}%"
                                    )
                                elif data["type"] == "error":
                                    print(f"\n[ERROR] {data['message']}")
                        except (json.JSONDecodeError, KeyError):
                            # Not JSON, treat as streaming text
                            print(chunk, end="", flush=True)
                            response += chunk
                    except asyncio.TimeoutError:
                        # No more messages for now
                        if response:
                            break

                print()  # New line after response

                # Show performance metrics in debug mode
                elapsed = time.time() - start_time
                self.response_times.append(elapsed)

                if self.debug_mode:
                    print(f"[DEBUG] Response time: {elapsed:.2f}s")

                # Track memory usage if enabled
                if self.memory_tracking:
                    process = psutil.Process(os.getpid())
                    memory_info = process.memory_info()
                    memory_mb = memory_info.rss / (1024 * 1024)
                    self.memory_samples.append(memory_mb)
                    print(
                        f"[MEMORY] Current: {memory_mb:.2f} MB, Avg: {sum(self.memory_samples) / len(self.memory_samples):.2f} MB"
                    )

        except Exception as e:
            logger.error(f"Error processing message: {str(e)}")
            print(f"Error: {str(e)}")

    async def run(self):
        """Run the validation CLI."""
        # Initialize components
        if not await self.initialize():
            logger.error("Failed to initialize components")
            return

        # Print header
        self.print_header()

        # Main loop
        running = True
        while running:
            try:
                # Check if monitoring is active and print a newline for better formatting
                if self.is_monitoring:
                    print()

                user_input = input("\nYou: ")

                # Check if this is a command
                if user_input.startswith("/"):
                    parts = user_input.split()
                    command = parts[0].lower()
                    args = parts[1:] if len(parts) > 1 else []

                    running = await self.handle_command(command, args)
                else:
                    # Process as a regular message
                    await self.process_message(user_input)

            except KeyboardInterrupt:
                print("\nExiting...")
                running = False
            except Exception as e:
                logger.error(f"Error: {str(e)}")
                import traceback

                traceback.print_exc()

        print("\nValidation CLI exited.")


async def main():
    """Main function."""
    parser = argparse.ArgumentParser(
        description="Interactive validation CLI for Milestone 4: API Layer and System Monitoring"
    )
    parser.add_argument("--interactive", action="store_true", help="Run in interactive mode")
    parser.add_argument("--api-url", default=DEFAULT_API_URL, help="API URL")
    parser.add_argument("--ws-url", default=DEFAULT_WS_URL, help="WebSocket URL")

    args = parser.parse_args()

    # Always run in interactive mode for now
    cli = ValidationCLI()
    cli.api_url = args.api_url
    cli.ws_url = args.ws_url
    await cli.run()


if __name__ == "__main__":
    asyncio.run(main())
