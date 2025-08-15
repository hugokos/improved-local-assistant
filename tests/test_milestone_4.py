"""
Test script for Milestone 4: API Layer and System Monitoring.

This script tests the FastAPI application, WebSocket connections,
and system monitoring features.
"""

import asyncio
import json
import logging
import sys
import time
import unittest
from concurrent.futures import ThreadPoolExecutor

import requests
import websockets

# Fix for Windows asyncio event loop issue
if sys.platform == "win32":
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

# Add project root to Python path for imports
project_root = Path(__file__).resolve().parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Default API URL
DEFAULT_API_URL = "http://localhost:8000"
DEFAULT_WS_URL = "ws://localhost:8000"


class TestMilestone4(unittest.TestCase):
    """Test cases for Milestone 4: API Layer and System Monitoring."""

    @classmethod
    def setUpClass(cls):
        """Set up test environment."""
        cls.api_url = DEFAULT_API_URL
        cls.ws_url = DEFAULT_WS_URL
        cls.session = requests.Session()

        # Check if server is running
        try:
            response = cls.session.get(f"{cls.api_url}/health")
            if response.status_code != 200:
                logger.error("API server is not running or not healthy")
                raise Exception("API server is not running or not healthy")
        except Exception as e:
            logger.error(f"Error connecting to API server: {str(e)}")
            raise

    def test_01_health_endpoint(self):
        """Test the health check endpoint."""
        logger.info("Testing health check endpoint...")
        response = self.session.get(f"{self.api_url}/health")

        self.assertEqual(response.status_code, 200)
        data = response.json()

        # Check required fields
        self.assertIn("status", data)
        self.assertIn("timestamp", data)
        self.assertIn("services", data)
        self.assertIn("models", data)
        self.assertIn("knowledge_graphs", data)
        self.assertIn("system", data)

        # Check service status
        self.assertIn("model_manager", data["services"])
        self.assertIn("kg_manager", data["services"])
        self.assertIn("conversation_manager", data["services"])
        self.assertIn("system_monitor", data["services"])

        logger.info("Health check endpoint test passed")

    def test_02_metrics_endpoint(self):
        """Test the metrics endpoint."""
        logger.info("Testing metrics endpoint...")
        response = self.session.get(f"{self.api_url}/metrics")

        self.assertEqual(response.status_code, 200)
        data = response.json()

        # Check required fields
        self.assertIn("timestamp", data)
        self.assertIn("resource_usage", data)
        self.assertIn("performance_metrics", data)

        # Check resource usage fields
        self.assertIn("cpu_percent", data["resource_usage"])
        self.assertIn("memory_percent", data["resource_usage"])
        self.assertIn("memory_used_gb", data["resource_usage"])
        self.assertIn("memory_total_gb", data["resource_usage"])

        logger.info("Metrics endpoint test passed")

    def test_03_system_info_endpoint(self):
        """Test the system info endpoint."""
        logger.info("Testing system info endpoint...")
        response = self.session.get(f"{self.api_url}/api/system/info")

        self.assertEqual(response.status_code, 200)
        data = response.json()

        # Check required fields
        self.assertIn("system_info", data)
        self.assertIn("resource_usage", data)
        self.assertIn("performance_metrics", data)
        self.assertIn("resource_limits", data)
        self.assertIn("health", data)

        logger.info("System info endpoint test passed")

    def test_04_chat_endpoint(self):
        """Test the chat endpoint."""
        logger.info("Testing chat endpoint...")

        payload = {"message": "Hello, how are you?", "session_id": None}

        response = self.session.post(f"{self.api_url}/api/chat", json=payload)

        self.assertEqual(response.status_code, 200)
        data = response.json()

        # Check required fields
        self.assertIn("response", data)
        self.assertIn("session_id", data)

        # Store session ID for future tests
        self.__class__.session_id = data["session_id"]

        logger.info("Chat endpoint test passed")

    def test_05_session_endpoints(self):
        """Test the session management endpoints."""
        logger.info("Testing session management endpoints...")

        # List sessions
        list_response = self.session.get(f"{self.api_url}/api/sessions")

        self.assertEqual(list_response.status_code, 200)
        list_data = list_response.json()

        # Check required fields
        self.assertIn("sessions", list_data)
        self.assertIn("count", list_data)

        # Get session info
        if hasattr(self.__class__, "session_id"):
            info_response = self.session.get(
                f"{self.api_url}/api/session/{self.__class__.session_id}"
            )

            self.assertEqual(info_response.status_code, 200)
            info_data = info_response.json()

            # Check required fields
            self.assertIn("session_id", info_data)
            self.assertIn("created_at", info_data)
            self.assertIn("updated_at", info_data)
            self.assertIn("message_count", info_data)

        logger.info("Session management endpoints test passed")

    def test_06_graph_endpoints(self):
        """Test the knowledge graph endpoints."""
        logger.info("Testing knowledge graph endpoints...")

        # List graphs
        list_response = self.session.get(f"{self.api_url}/api/graphs")

        self.assertEqual(list_response.status_code, 200)
        list_data = list_response.json()

        # Check required fields
        self.assertIn("graphs", list_data)
        self.assertIn("total_graphs", list_data)
        self.assertIn("total_nodes", list_data)
        self.assertIn("total_edges", list_data)

        # Query graph
        query_payload = {
            "query": "What is a knowledge graph?",
            "session_id": getattr(self.__class__, "session_id", None),
        }

        query_response = self.session.post(f"{self.api_url}/api/graph/query", json=query_payload)

        self.assertEqual(query_response.status_code, 200)
        query_data = query_response.json()

        # Check required fields
        self.assertIn("response", query_data)
        self.assertIn("metadata", query_data)

        # Visualize graph
        viz_response = self.session.get(f"{self.api_url}/api/graph/visualize")

        self.assertEqual(viz_response.status_code, 200)
        self.assertIn("<html", viz_response.text.lower())

        logger.info("Knowledge graph endpoints test passed")

    async def _test_websocket_connection(self):
        """Test WebSocket connection and message streaming."""
        logger.info("Testing WebSocket connection...")

        session_id = getattr(self.__class__, "session_id", f"test_{int(time.time())}")

        try:
            async with websockets.connect(f"{self.ws_url}/ws/{session_id}") as websocket:
                logger.info("WebSocket connected successfully")

                # Send a test message
                test_message = "Hello, can you tell me about knowledge graphs?"
                logger.info(f"Sending message: '{test_message}'")
                await websocket.send(test_message)

                # Receive streaming response
                response = ""
                received_graph = False

                # Wait for up to 30 seconds for a complete response
                start_time = time.time()
                while time.time() - start_time < 30:
                    try:
                        message = await asyncio.wait_for(websocket.recv(), timeout=1.0)

                        # Try to parse as JSON
                        try:
                            data = json.loads(message)
                            if "type" in data and data["type"] == "knowledge_graph":
                                logger.info("Received knowledge graph data")
                                received_graph = True
                                break
                        except (json.JSONDecodeError, KeyError):
                            # Not JSON, treat as streaming text
                            response += message
                    except asyncio.TimeoutError:
                        # No more messages for now, continue waiting
                        if response and received_graph:
                            break

                # Verify we got a response
                self.assertTrue(len(response) > 0, "No text response received from WebSocket")

                logger.info("WebSocket test completed successfully")
                return True
        except Exception as e:
            logger.error(f"Error in WebSocket test: {str(e)}")
            raise

    async def _test_monitor_websocket(self):
        """Test the monitoring WebSocket endpoint."""
        logger.info("Testing monitoring WebSocket...")

        try:
            async with websockets.connect(f"{self.ws_url}/ws/monitor") as websocket:
                logger.info("Monitoring WebSocket connected successfully")

                # Receive initial status
                message = await asyncio.wait_for(websocket.recv(), timeout=5.0)
                data = json.loads(message)

                # Check required fields
                self.assertIn("type", data)
                self.assertEqual(data["type"], "system_status")
                self.assertIn("resource_usage", data)
                self.assertIn("performance_metrics", data)

                # Wait for another update
                message = await asyncio.wait_for(websocket.recv(), timeout=5.0)
                data = json.loads(message)

                # Check required fields again
                self.assertIn("type", data)
                self.assertEqual(data["type"], "system_status")

                logger.info("Monitoring WebSocket test completed successfully")
                return True
        except Exception as e:
            logger.error(f"Error in monitoring WebSocket test: {str(e)}")
            raise

    async def _test_concurrent_websockets(self, num_connections=3):
        """Test multiple concurrent WebSocket connections."""
        logger.info(f"Testing {num_connections} concurrent WebSocket connections...")

        async def connect_and_chat(session_id):
            try:
                async with websockets.connect(f"{self.ws_url}/ws/{session_id}") as websocket:
                    # Send a message
                    await websocket.send(f"Hello from session {session_id}")

                    # Wait for response
                    start_time = time.time()
                    while time.time() - start_time < 10:
                        try:
                            await asyncio.wait_for(websocket.recv(), timeout=1.0)
                        except asyncio.TimeoutError:
                            continue

                    return True
            except Exception as e:
                logger.error(f"Error in connection {session_id}: {str(e)}")
                return False

        # Create tasks for multiple connections
        tasks = []
        for i in range(num_connections):
            session_id = f"test_concurrent_{i}_{int(time.time())}"
            tasks.append(connect_and_chat(session_id))

        # Run all tasks concurrently
        results = await asyncio.gather(*tasks, return_exceptions=True)

        # Check results
        success_count = sum(1 for r in results if r is True)
        logger.info(f"{success_count}/{num_connections} concurrent connections successful")

        # At least half of the connections should succeed
        self.assertGreaterEqual(success_count, num_connections // 2)

        return success_count == num_connections

    def test_07_websocket_connection(self):
        """Test WebSocket connection and message streaming."""
        result = asyncio.run(self._test_websocket_connection())
        self.assertTrue(result)

    def test_08_monitor_websocket(self):
        """Test the monitoring WebSocket endpoint."""
        result = asyncio.run(self._test_monitor_websocket())
        self.assertTrue(result)

    def test_09_concurrent_websockets(self):
        """Test multiple concurrent WebSocket connections."""
        result = asyncio.run(self._test_concurrent_websockets(3))
        self.assertTrue(result)

    def test_10_error_handling(self):
        """Test error handling in API endpoints."""
        logger.info("Testing error handling...")

        # Test invalid session ID
        invalid_session = self.session.get(f"{self.api_url}/api/session/invalid_session_id")
        self.assertEqual(invalid_session.status_code, 404)

        # Test invalid graph ID
        invalid_graph = self.session.get(f"{self.api_url}/api/graph/invalid_graph_id/stats")
        self.assertEqual(invalid_graph.status_code, 404)

        # Test invalid JSON payload
        invalid_json = requests.post(
            f"{self.api_url}/api/chat",
            data="invalid json",
            headers={"Content-Type": "application/json"},
        )
        self.assertEqual(invalid_json.status_code, 422)  # Unprocessable Entity

        logger.info("Error handling test passed")

    def test_11_performance(self):
        """Test API performance under load."""
        logger.info("Testing API performance...")

        # Number of requests to send
        num_requests = 10

        # Test endpoint to use
        endpoint = f"{self.api_url}/health"

        # Send requests concurrently
        with ThreadPoolExecutor(max_workers=5) as executor:
            start_time = time.time()
            futures = [executor.submit(self.session.get, endpoint) for _ in range(num_requests)]
            responses = [future.result() for future in futures]
            end_time = time.time()

        # Calculate statistics
        success_count = sum(1 for r in responses if r.status_code == 200)
        avg_time = (end_time - start_time) / num_requests

        logger.info(f"Performance test: {success_count}/{num_requests} successful requests")
        logger.info(f"Average request time: {avg_time:.4f} seconds")

        # All requests should succeed
        self.assertEqual(success_count, num_requests)

        # Average time should be reasonable (adjust based on your hardware)
        self.assertLess(avg_time, 1.0)  # Less than 1 second per request on average

        logger.info("Performance test passed")


if __name__ == "__main__":
    unittest.main()
