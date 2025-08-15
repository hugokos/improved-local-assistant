#!/usr/bin/env python3
"""
Web Interface Test Suite

This module contains tests for the web interface components of the improved local AI assistant.
"""

import asyncio
import http.server
import json
import os
import socketserver
import sys
import threading
import time
import unittest
from pathlib import Path

import requests
import websockets

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))


class TestWebInterface(unittest.TestCase):
    """Test suite for the web interface"""

    @classmethod
    def setUpClass(cls):
        """Set up the test environment"""
        cls.static_dir = Path(__file__).resolve().parent.parent.parent / "app" / "static"
        cls.http_port = 8080
        cls.ws_port = 8081

        # Start HTTP server in a separate thread
        cls.http_server = cls._start_http_server(cls.http_port)

        # Start WebSocket server in a separate thread
        cls.ws_thread = threading.Thread(
            target=lambda: asyncio.run(cls._run_mock_websocket_server(cls.ws_port)), daemon=True
        )
        cls.ws_thread.start()

        # Wait for servers to start
        time.sleep(1)

    @classmethod
    def tearDownClass(cls):
        """Clean up the test environment"""
        cls.http_server.shutdown()
        cls.http_server.server_close()

    @classmethod
    def _start_http_server(cls, port):
        """Start an HTTP server for serving static files"""

        class Handler(http.server.SimpleHTTPRequestHandler):
            def __init__(self, *args, **kwargs):
                os.chdir(str(cls.static_dir))
                super().__init__(*args, **kwargs)

            def log_message(self, format, *args):
                # Suppress log messages
                pass

        httpd = socketserver.TCPServer(("", port), Handler)
        server_thread = threading.Thread(target=httpd.serve_forever)
        server_thread.daemon = True
        server_thread.start()
        return httpd

    @classmethod
    async def _run_mock_websocket_server(cls, port):
        """Run a mock WebSocket server for testing"""

        async def handler(websocket, path):
            try:
                if path.startswith("/ws/monitor"):
                    # Send mock monitoring data periodically
                    while True:
                        data = {
                            "type": "system_status",
                            "resource_usage": {"cpu_percent": 25, "memory_percent": 40},
                            "response_time": 150,
                        }
                        await websocket.send(json.dumps(data))
                        await asyncio.sleep(1)
                else:
                    # Regular chat websocket
                    await websocket.send(
                        json.dumps(
                            {
                                "type": "system",
                                "message": "Connected to test WebSocket server",
                                "session_id": path.split("/")[-1],
                            }
                        )
                    )

                    # Send available graphs
                    await websocket.send(
                        json.dumps(
                            {
                                "type": "available_graphs",
                                "graphs": [
                                    {"id": "graph1", "name": "General Knowledge"},
                                    {"id": "graph2", "name": "Programming"},
                                    {"id": "graph3", "name": "Science"},
                                ],
                            }
                        )
                    )

                    while True:
                        try:
                            message = await websocket.recv()
                            print(f"Received message: {message}")

                            # Check if it's a JSON message
                            try:
                                data = json.loads(message)
                                if data.get("type") == "settings_update":
                                    await websocket.send(
                                        json.dumps(
                                            {
                                                "type": "system",
                                                "message": "Settings updated successfully",
                                            }
                                        )
                                    )
                                    continue
                                elif data.get("type") == "load_graphs":
                                    await websocket.send(
                                        json.dumps(
                                            {
                                                "type": "system",
                                                "message": f"Loaded graphs: {', '.join(data.get('graph_ids', []))}",
                                            }
                                        )
                                    )
                                    continue
                                elif data.get("type") == "unload_graphs":
                                    await websocket.send(
                                        json.dumps(
                                            {
                                                "type": "system",
                                                "message": f"Unloaded graphs: {', '.join(data.get('graph_ids', []))}",
                                            }
                                        )
                                    )
                                    continue
                            except (json.JSONDecodeError, KeyError):
                                pass

                            # Send typing indicator
                            await websocket.send(json.dumps({"type": "typing", "status": "start"}))

                            # Stream response
                            response = "This is a test response from the mock WebSocket server."
                            for char in response:
                                await websocket.send(char)
                                await asyncio.sleep(0.05)

                            # Stop typing
                            await websocket.send(json.dumps({"type": "typing", "status": "stop"}))

                            # Send mock knowledge graph data
                            await asyncio.sleep(1)
                            await websocket.send(
                                json.dumps(
                                    {
                                        "type": "knowledge_graph",
                                        "data": "<div class='mock-graph'>Mock Knowledge Graph Visualization</div>",
                                    }
                                )
                            )

                        except websockets.exceptions.ConnectionClosed:
                            break
            except Exception as e:
                print(f"WebSocket error: {e}")

        async with websockets.serve(handler, "localhost", port):
            await asyncio.Future()  # Run forever

    def test_static_files_exist(self):
        """Test that all required static files exist"""
        required_files = ["index.html", "style.css", "script.js"]
        for file in required_files:
            file_path = self.static_dir / file
            self.assertTrue(file_path.exists(), f"File {file} does not exist")

    def test_html_structure(self):
        """Test that the HTML structure contains all required elements"""
        with open(self.static_dir / "index.html") as f:
            html_content = f.read()

        # Check for main containers
        self.assertIn('<div class="chat-container">', html_content)
        self.assertIn('<div class="sidebar" id="kgSidebar">', html_content)
        self.assertIn('<div class="monitoring-panel">', html_content)
        self.assertIn('<div id="settingsModal" class="modal">', html_content)

        # Check for responsive meta tag
        self.assertIn(
            '<meta name="viewport" content="width=device-width, initial-scale=1.0">', html_content
        )

    def test_css_responsive_rules(self):
        """Test that the CSS contains responsive design rules"""
        with open(self.static_dir / "style.css") as f:
            css_content = f.read()

        # Check for media queries
        self.assertIn("@media (max-width:", css_content)

        # Check for flex layout
        self.assertIn("display: flex;", css_content)

        # Check for responsive sidebar
        self.assertIn(".sidebar.collapsed", css_content)

    def test_javascript_functionality(self):
        """Test that the JavaScript file contains all required functionality"""
        with open(self.static_dir / "script.js") as f:
            js_content = f.read()

        # Check for WebSocket handling
        self.assertIn("WebSocket(", js_content)

        # Check for message handling
        self.assertIn("sendMessage()", js_content)

        # Check for settings functionality
        self.assertIn("saveSettings()", js_content)

        # Check for graph functionality
        self.assertIn("updateKnowledgeGraph(", js_content)

    def test_http_server_response(self):
        """Test that the HTTP server responds correctly"""
        response = requests.get(f"http://localhost:{self.http_port}/index.html")
        self.assertEqual(response.status_code, 200)
        self.assertIn("<!DOCTYPE html>", response.text)


def run_tests():
    """Run the test suite"""
    unittest.main()


if __name__ == "__main__":
    run_tests()
