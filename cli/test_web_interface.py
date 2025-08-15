#!/usr/bin/env python3
"""
Web Interface Testing Tool

This script provides functionality to test the web interface components
of the improved local AI assistant.

Usage:
    python cli/test_web_interface.py --serve-static  # Serve static files
    python cli/test_web_interface.py --test-chat     # Test chat interface
    python cli/test_web_interface.py --test-sidebar  # Test graph sidebar
    python cli/test_web_interface.py --test-monitoring  # Test monitoring
    python cli/test_web_interface.py --test-settings  # Test settings panel
"""

import argparse
import asyncio
import http.server
import json
import os
import random
import socketserver
import sys
import threading
import webbrowser
from pathlib import Path

import websockets

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


class StaticFileHandler(http.server.SimpleHTTPRequestHandler):
    """Handler for serving static files"""

    def __init__(self, *args, **kwargs):
        static_dir = str(Path(__file__).resolve().parent.parent / "app" / "static")
        os.chdir(static_dir)
        super().__init__(*args, **kwargs)

    def log_message(self, format, *args):
        # Suppress log messages
        pass


def serve_static_files(port=8000):
    """Serve static files from the app/static directory"""
    handler = StaticFileHandler

    with socketserver.TCPServer(("", port), handler) as httpd:
        print(f"Serving static files at http://localhost:{port}")
        print("Press Ctrl+C to stop the server")
        try:
            httpd.serve_forever()
        except KeyboardInterrupt:
            print("\nServer stopped")


async def mock_websocket_server(port=8001):
    """Run a mock WebSocket server for testing"""

    connected_clients = set()

    async def handler(websocket, path):
        connected_clients.add(websocket)
        print(f"Client connected: {path}")

        try:
            if path.startswith("/ws/monitor"):
                # Send mock monitoring data periodically
                while True:
                    data = {
                        "type": "system_status",
                        "resource_usage": {
                            "cpu_percent": random.randint(5, 95),
                            "memory_percent": random.randint(10, 80),
                        },
                    }
                    await websocket.send(json.dumps(data))
                    await asyncio.sleep(2)
            else:
                # Regular chat websocket
                await websocket.send(
                    json.dumps({"type": "system", "message": "Connected to test WebSocket server"})
                )

                while True:
                    try:
                        message = await websocket.recv()
                        print(f"Received message: {message}")

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
        finally:
            connected_clients.remove(websocket)
            print("Client disconnected")

    async with websockets.serve(handler, "localhost", port):
        print(f"Mock WebSocket server running at ws://localhost:{port}")
        await asyncio.Future()  # Run forever


def test_chat_interface(port=8000, ws_port=8001):
    """Test the chat interface with WebSocket functionality"""
    # Start the WebSocket server in a separate thread
    ws_thread = threading.Thread(
        target=lambda: asyncio.run(mock_websocket_server(ws_port)), daemon=True
    )
    ws_thread.start()

    # Modify the script.js file to use the mock WebSocket server
    script_path = Path(__file__).resolve().parent.parent / "app" / "static" / "script.js"
    with open(script_path) as f:
        script_content = f.read()

    # Replace WebSocket connection URL
    modified_script = script_content.replace(
        "ws://${window.location.host}/ws/", f"ws://localhost:{ws_port}/ws/"
    )

    # Save modified script to a temporary file
    temp_script_path = Path(__file__).resolve().parent.parent / "app" / "static" / "temp_script.js"
    with open(temp_script_path, "w") as f:
        f.write(modified_script)

    # Modify index.html to use the temporary script
    index_path = Path(__file__).resolve().parent.parent / "app" / "static" / "index.html"
    with open(index_path) as f:
        index_content = f.read()

    modified_index = index_content.replace(
        '<script src="script.js"></script>', '<script src="temp_script.js"></script>'
    )

    temp_index_path = Path(__file__).resolve().parent.parent / "app" / "static" / "temp_index.html"
    with open(temp_index_path, "w") as f:
        f.write(modified_index)

    # Start the HTTP server
    print("Starting test server for chat interface...")
    print(f"Open http://localhost:{port}/temp_index.html in your browser")

    # Open the browser
    webbrowser.open(f"http://localhost:{port}/temp_index.html")

    # Serve the files
    serve_static_files(port)

    # Clean up temporary files
    temp_script_path.unlink(missing_ok=True)
    temp_index_path.unlink(missing_ok=True)


def test_sidebar():
    """Test the knowledge graph sidebar"""
    # Create a test HTML file with mock graph data
    test_html = """
    <!DOCTYPE html>
    <html>
    <head>
        <title>Knowledge Graph Sidebar Test</title>
        <link rel="stylesheet" href="style.css">
        <style>
            body { padding: 20px; }
            .test-controls { margin-bottom: 20px; }
            button { margin-right: 10px; padding: 8px 16px; }
            .mock-graph { 
                height: 400px; 
                border: 1px solid #ddd; 
                display: flex; 
                align-items: center; 
                justify-content: center;
                margin-top: 20px;
            }
        </style>
    </head>
    <body>
        <h1>Knowledge Graph Sidebar Test</h1>
        
        <div class="test-controls">
            <button onclick="toggleSidebar()">Toggle Sidebar</button>
            <button onclick="loadMockGraph()">Load Mock Graph</button>
        </div>
        
        <div class="main-content">
            <div class="sidebar" id="kgSidebar">
                <div class="sidebar-header">
                    <h3>Knowledge Graph</h3>
                    <button onclick="toggleSidebar()">Toggle</button>
                </div>
                <div class="graph-container" id="graphContainer">
                    <p>No graph data available</p>
                </div>
            </div>
        </div>
        
        <script>
            function toggleSidebar() {
                const sidebar = document.getElementById('kgSidebar');
                sidebar.classList.toggle('collapsed');
            }
            
            function loadMockGraph() {
                const container = document.getElementById('graphContainer');
                container.innerHTML = '<div class="mock-graph">Mock Knowledge Graph Visualization</div>';
            }
        </script>
    </body>
    </html>
    """

    # Save the test HTML
    test_path = Path(__file__).resolve().parent.parent / "app" / "static" / "test_sidebar.html"
    with open(test_path, "w") as f:
        f.write(test_html)

    # Start the server and open the browser
    port = 8000
    print(f"Testing sidebar at http://localhost:{port}/test_sidebar.html")
    webbrowser.open(f"http://localhost:{port}/test_sidebar.html")

    # Serve the files
    serve_static_files(port)

    # Clean up
    test_path.unlink(missing_ok=True)


def test_monitoring():
    """Test the system monitoring dashboard"""
    # Create a test HTML file with mock monitoring data
    test_html = """
    <!DOCTYPE html>
    <html>
    <head>
        <title>System Monitoring Test</title>
        <link rel="stylesheet" href="style.css">
        <style>
            body { padding: 20px; }
            .test-controls { margin-bottom: 20px; }
            button { margin-right: 10px; padding: 8px 16px; }
            .monitoring-panel {
                display: flex;
                flex-wrap: wrap;
                gap: 20px;
                margin-top: 20px;
                padding: 15px;
                background: white;
                border-radius: 8px;
                box-shadow: 0 2px 4px rgba(0,0,0,0.1);
            }
            .metric {
                display: flex;
                gap: 10px;
                align-items: center;
            }
            .gauge {
                width: 100px;
                height: 100px;
                border-radius: 50%;
                border: 10px solid #eee;
                position: relative;
                margin: 10px;
            }
            .gauge-fill {
                position: absolute;
                top: 0;
                left: 0;
                width: 100%;
                height: 100%;
                border-radius: 50%;
                clip: rect(0, 50px, 100px, 0);
                background: #1976d2;
                transform-origin: center;
            }
            .gauge-value {
                position: absolute;
                top: 50%;
                left: 50%;
                transform: translate(-50%, -50%);
                font-size: 16px;
                font-weight: bold;
            }
        </style>
    </head>
    <body>
        <h1>System Monitoring Test</h1>
        
        <div class="test-controls">
            <button onclick="updateRandomMetrics()">Update Random Metrics</button>
            <button onclick="simulateHighLoad()">Simulate High Load</button>
            <button onclick="simulateLowLoad()">Simulate Low Load</button>
        </div>
        
        <div class="monitoring-panel">
            <div class="metric">
                <span>CPU Usage:</span>
                <span id="cpuUsage">0%</span>
                <div class="gauge">
                    <div class="gauge-fill" id="cpuGauge" style="transform: rotate(0deg);"></div>
                    <div class="gauge-value" id="cpuValue">0%</div>
                </div>
            </div>
            <div class="metric">
                <span>Memory Usage:</span>
                <span id="memoryUsage">0%</span>
                <div class="gauge">
                    <div class="gauge-fill" id="memoryGauge" style="transform: rotate(0deg);"></div>
                    <div class="gauge-value" id="memoryValue">0%</div>
                </div>
            </div>
        </div>
        
        <div class="monitoring-panel">
            <div class="metric">
                <span>Model Status:</span>
                <span id="modelStatus">OK</span>
            </div>
            <div class="metric">
                <span>Response Time:</span>
                <span id="responseTime">0ms</span>
            </div>
        </div>
        
        <script>
            function updateMetrics(cpu, memory, responseTime) {
                // Update text values
                document.getElementById('cpuUsage').textContent = cpu + '%';
                document.getElementById('memoryUsage').textContent = memory + '%';
                document.getElementById('responseTime').textContent = responseTime + 'ms';
                document.getElementById('cpuValue').textContent = cpu + '%';
                document.getElementById('memoryValue').textContent = memory + '%';
                
                // Update gauges
                const cpuDegrees = (cpu / 100) * 180;
                const memoryDegrees = (memory / 100) * 180;
                
                document.getElementById('cpuGauge').style.transform = `rotate(${cpuDegrees}deg)`;
                document.getElementById('memoryGauge').style.transform = `rotate(${memoryDegrees}deg)`;
                
                // Update status based on load
                if (cpu > 80 || memory > 80) {
                    document.getElementById('modelStatus').textContent = 'High Load';
                    document.getElementById('modelStatus').style.color = 'red';
                } else if (cpu > 50 || memory > 50) {
                    document.getElementById('modelStatus').textContent = 'Moderate Load';
                    document.getElementById('modelStatus').style.color = 'orange';
                } else {
                    document.getElementById('modelStatus').textContent = 'OK';
                    document.getElementById('modelStatus').style.color = 'green';
                }
            }
            
            function updateRandomMetrics() {
                const cpu = Math.floor(Math.random() * 100);
                const memory = Math.floor(Math.random() * 100);
                const responseTime = Math.floor(Math.random() * 2000);
                updateMetrics(cpu, memory, responseTime);
            }
            
            function simulateHighLoad() {
                updateMetrics(85, 90, 1500);
            }
            
            function simulateLowLoad() {
                updateMetrics(15, 30, 200);
            }
            
            // Update metrics every 2 seconds
            setInterval(updateRandomMetrics, 2000);
            
            // Initial update
            updateRandomMetrics();
        </script>
    </body>
    </html>
    """

    # Save the test HTML
    test_path = Path(__file__).resolve().parent.parent / "app" / "static" / "test_monitoring.html"
    with open(test_path, "w") as f:
        f.write(test_html)

    # Start the server and open the browser
    port = 8000
    print(f"Testing monitoring at http://localhost:{port}/test_monitoring.html")
    webbrowser.open(f"http://localhost:{port}/test_monitoring.html")

    # Serve the files
    serve_static_files(port)

    # Clean up
    test_path.unlink(missing_ok=True)


def test_settings():
    """Test the settings and configuration interface"""
    # Create a test HTML file with mock settings interface
    test_html = """
    <!DOCTYPE html>
    <html>
    <head>
        <title>Settings Interface Test</title>
        <link rel="stylesheet" href="style.css">
        <style>
            body { padding: 20px; }
            .settings-container {
                max-width: 800px;
                margin: 0 auto;
                background: white;
                border-radius: 8px;
                box-shadow: 0 2px 4px rgba(0,0,0,0.1);
                padding: 20px;
            }
            .settings-section {
                margin-bottom: 30px;
                border-bottom: 1px solid #eee;
                padding-bottom: 20px;
            }
            .settings-section:last-child {
                border-bottom: none;
            }
            .settings-row {
                display: flex;
                justify-content: space-between;
                align-items: center;
                margin-bottom: 15px;
            }
            .settings-controls {
                display: flex;
                gap: 10px;
            }
            select, input {
                padding: 8px;
                border: 1px solid #ddd;
                border-radius: 4px;
            }
            button {
                padding: 8px 16px;
                background: #1976d2;
                color: white;
                border: none;
                border-radius: 4px;
                cursor: pointer;
            }
            .status-message {
                margin-top: 20px;
                padding: 10px;
                border-radius: 4px;
                background: #e3f2fd;
                display: none;
            }
        </style>
    </head>
    <body>
        <h1>Settings Interface Test</h1>
        
        <div class="settings-container">
            <div class="settings-section">
                <h2>Model Configuration</h2>
                
                <div class="settings-row">
                    <span>Conversation Model:</span>
                    <div class="settings-controls">
                        <select id="conversationModel">
                            <option value="hermes:3b">Hermes 3:3B</option>
                            <option value="llama2:7b">Llama2 7B</option>
                            <option value="mistral:7b">Mistral 7B</option>
                        </select>
                        <button onclick="saveModelSettings('conversation')">Apply</button>
                    </div>
                </div>
                
                <div class="settings-row">
                    <span>Background Model:</span>
                    <div class="settings-controls">
                        <select id="backgroundModel">
                            <option value="tinyllama">TinyLlama</option>
                            <option value="phi:1.5b">Phi 1.5B</option>
                        </select>
                        <button onclick="saveModelSettings('background')">Apply</button>
                    </div>
                </div>
                
                <div class="settings-row">
                    <span>Temperature:</span>
                    <div class="settings-controls">
                        <input type="range" id="temperature" min="0" max="1" step="0.1" value="0.7">
                        <span id="temperatureValue">0.7</span>
                        <button onclick="saveModelSettings('temperature')">Apply</button>
                    </div>
                </div>
            </div>
            
            <div class="settings-section">
                <h2>Knowledge Graph Management</h2>
                
                <div class="settings-row">
                    <span>Available Graphs:</span>
                    <div class="settings-controls">
                        <select id="graphSelect" multiple size="3">
                            <option value="graph1">General Knowledge</option>
                            <option value="graph2">Programming</option>
                            <option value="graph3">Science</option>
                        </select>
                    </div>
                </div>
                
                <div class="settings-row">
                    <span>Graph Operations:</span>
                    <div class="settings-controls">
                        <button onclick="loadGraph()">Load Selected</button>
                        <button onclick="unloadGraph()">Unload Selected</button>
                        <button onclick="importGraph()">Import New</button>
                    </div>
                </div>
            </div>
            
            <div class="settings-section">
                <h2>System Performance</h2>
                
                <div class="settings-row">
                    <span>Max Memory Usage:</span>
                    <div class="settings-controls">
                        <input type="range" id="memoryLimit" min="2" max="12" step="1" value="8">
                        <span id="memoryLimitValue">8 GB</span>
                        <button onclick="saveSystemSettings('memory')">Apply</button>
                    </div>
                </div>
                
                <div class="settings-row">
                    <span>Concurrent Operations:</span>
                    <div class="settings-controls">
                        <input type="number" id="concurrentOps" min="1" max="4" value="2">
                        <button onclick="saveSystemSettings('concurrent')">Apply</button>
                    </div>
                </div>
            </div>
            
            <div class="settings-section">
                <h2>Configuration Import/Export</h2>
                
                <div class="settings-row">
                    <span>Configuration File:</span>
                    <div class="settings-controls">
                        <button onclick="exportConfig()">Export</button>
                        <button onclick="importConfig()">Import</button>
                    </div>
                </div>
            </div>
            
            <div class="status-message" id="statusMessage"></div>
        </div>
        
        <script>
            // Update temperature value display
            document.getElementById('temperature').addEventListener('input', function() {
                document.getElementById('temperatureValue').textContent = this.value;
            });
            
            // Update memory limit value display
            document.getElementById('memoryLimit').addEventListener('input', function() {
                document.getElementById('memoryLimitValue').textContent = this.value + ' GB';
            });
            
            function showStatus(message, isError = false) {
                const statusElement = document.getElementById('statusMessage');
                statusElement.textContent = message;
                statusElement.style.display = 'block';
                statusElement.style.background = isError ? '#ffebee' : '#e8f5e9';
                statusElement.style.color = isError ? '#c62828' : '#2e7d32';
                
                // Hide after 3 seconds
                setTimeout(() => {
                    statusElement.style.display = 'none';
                }, 3000);
            }
            
            function saveModelSettings(type) {
                switch(type) {
                    case 'conversation':
                        const convModel = document.getElementById('conversationModel').value;
                        showStatus(`Conversation model changed to ${convModel}`);
                        break;
                    case 'background':
                        const bgModel = document.getElementById('backgroundModel').value;
                        showStatus(`Background model changed to ${bgModel}`);
                        break;
                    case 'temperature':
                        const temp = document.getElementById('temperature').value;
                        showStatus(`Temperature set to ${temp}`);
                        break;
                }
            }
            
            function saveSystemSettings(type) {
                switch(type) {
                    case 'memory':
                        const memory = document.getElementById('memoryLimit').value;
                        showStatus(`Memory limit set to ${memory} GB`);
                        break;
                    case 'concurrent':
                        const concurrent = document.getElementById('concurrentOps').value;
                        showStatus(`Concurrent operations set to ${concurrent}`);
                        break;
                }
            }
            
            function loadGraph() {
                const select = document.getElementById('graphSelect');
                const selected = Array.from(select.selectedOptions).map(option => option.text);
                
                if (selected.length === 0) {
                    showStatus('No graphs selected', true);
                } else {
                    showStatus(`Loaded graphs: ${selected.join(', ')}`);
                }
            }
            
            function unloadGraph() {
                const select = document.getElementById('graphSelect');
                const selected = Array.from(select.selectedOptions).map(option => option.text);
                
                if (selected.length === 0) {
                    showStatus('No graphs selected', true);
                } else {
                    showStatus(`Unloaded graphs: ${selected.join(', ')}`);
                }
            }
            
            function importGraph() {
                showStatus('Graph import functionality would open a file dialog');
            }
            
            function exportConfig() {
                showStatus('Configuration exported to config.json');
            }
            
            function importConfig() {
                showStatus('Configuration import functionality would open a file dialog');
            }
        </script>
    </body>
    </html>
    """

    # Save the test HTML
    test_path = Path(__file__).resolve().parent.parent / "app" / "static" / "test_settings.html"
    with open(test_path, "w") as f:
        f.write(test_html)

    # Start the server and open the browser
    port = 8000
    print(f"Testing settings at http://localhost:{port}/test_settings.html")
    webbrowser.open(f"http://localhost:{port}/test_settings.html")

    # Serve the files
    serve_static_files(port)

    # Clean up
    test_path.unlink(missing_ok=True)


def main():
    parser = argparse.ArgumentParser(description="Web Interface Testing Tool")
    parser.add_argument("--serve-static", action="store_true", help="Serve static files")
    parser.add_argument("--test-chat", action="store_true", help="Test chat interface")
    parser.add_argument("--test-sidebar", action="store_true", help="Test graph sidebar")
    parser.add_argument("--test-monitoring", action="store_true", help="Test monitoring")
    parser.add_argument("--test-settings", action="store_true", help="Test settings panel")
    parser.add_argument("--port", type=int, default=8000, help="Port for HTTP server")

    args = parser.parse_args()

    if args.serve_static:
        serve_static_files(args.port)
    elif args.test_chat:
        test_chat_interface(args.port)
    elif args.test_sidebar:
        test_sidebar()
    elif args.test_monitoring:
        test_monitoring()
    elif args.test_settings:
        test_settings()
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
