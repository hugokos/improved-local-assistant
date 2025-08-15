#!/usr/bin/env python3
"""
Milestone 5 Validation Tool

This script provides an interactive validation interface for the web interface
components of the improved local AI assistant.

Usage:
    python cli/validate_milestone_5.py --interactive
"""

import argparse
import http.server
import os
import socketserver
import sys
import unittest
import webbrowser
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


class TestWebInterface(unittest.TestCase):
    """Test suite for the web interface"""

    @classmethod
    def setUpClass(cls):
        """Set up the test environment"""
        cls.static_dir = Path(__file__).resolve().parent.parent / "app" / "static"

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


class SimpleHTTPRequestHandler(http.server.SimpleHTTPRequestHandler):
    """Simple HTTP request handler for serving static files"""

    def __init__(self, *args, static_dir=None, **kwargs):
        self.static_dir = static_dir
        super().__init__(*args, **kwargs)

    def log_message(self, format, *args):
        """Suppress log messages"""
        pass


def serve_static_files(port=8000):
    """Serve static files from the app/static directory"""
    static_dir = Path(__file__).resolve().parent.parent / "app" / "static"

    # Change to the static directory
    os.chdir(str(static_dir))

    # Create server
    handler = SimpleHTTPRequestHandler
    httpd = socketserver.TCPServer(("", port), handler)

    print(f"Serving static files at http://localhost:{port}")
    print("Press Ctrl+C to stop the server")

    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\nServer stopped")
    finally:
        httpd.server_close()


def run_automated_tests():
    """Run automated tests for the web interface"""
    print("\n=== Running Automated Tests ===\n")

    # Run tests
    test_loader = unittest.TestLoader()
    test_suite = test_loader.loadTestsFromTestCase(TestWebInterface)
    test_result = unittest.TextTestRunner().run(test_suite)

    # Print results
    print(f"Tests run: {test_result.testsRun}")
    print(f"Failures: {len(test_result.failures)}")
    print(f"Errors: {len(test_result.errors)}")

    if test_result.wasSuccessful():
        print("\n✅ All tests passed!")
    else:
        print("\n❌ Some tests failed:")
        for failure in test_result.failures:
            print(f"  - {failure[0]}")
        for error in test_result.errors:
            print(f"  - {error[0]}")

    return test_result.wasSuccessful()


def validate_responsive_design():
    """Validate responsive design across different screen sizes"""
    print("\n=== Validating Responsive Design ===\n")

    # Check CSS for media queries
    css_path = Path(__file__).resolve().parent.parent / "app" / "static" / "style.css"
    with open(css_path) as f:
        css_content = f.read()

    media_queries = css_content.count("@media")

    print(f"Found {media_queries} media queries in CSS")

    # Check for mobile-specific styles
    mobile_styles = "@media (max-width: 768px)" in css_content
    small_screen_styles = "@media (max-width: 480px)" in css_content

    if mobile_styles:
        print("✅ Mobile styles found (@media (max-width: 768px))")
    else:
        print("❌ No mobile styles found")

    if small_screen_styles:
        print("✅ Small screen styles found (@media (max-width: 480px))")
    else:
        print("❌ No small screen styles found")

    # Check for flex layout
    flex_layout = "display: flex;" in css_content
    if flex_layout:
        print("✅ Flex layout found")
    else:
        print("❌ No flex layout found")

    # Check for responsive sidebar
    responsive_sidebar = ".sidebar.collapsed" in css_content
    if responsive_sidebar:
        print("✅ Responsive sidebar found")
    else:
        print("❌ No responsive sidebar found")

    return mobile_styles and flex_layout and responsive_sidebar


def validate_ui_components():
    """Validate UI components"""
    print("\n=== Validating UI Components ===\n")

    # Check HTML for required components
    html_path = Path(__file__).resolve().parent.parent / "app" / "static" / "index.html"
    with open(html_path) as f:
        html_content = f.read()

    components = {
        "Chat Container": '<div class="chat-container">',
        "Knowledge Graph Sidebar": '<div class="sidebar" id="kgSidebar">',
        "Monitoring Panel": '<div class="monitoring-panel">',
        "Settings Modal": '<div id="settingsModal" class="modal">',
        "CPU Usage Indicator": 'id="cpuUsage"',
        "Memory Usage Indicator": 'id="memoryUsage"',
        "Response Time Indicator": 'id="responseTime"',
        "Progress Bars": 'class="progress-bar"',
        "Graph Container": 'id="graphContainer"',
        "Message Input": 'id="messageInput"',
        "Send Button": 'id="sendButton"',
    }

    all_found = True
    for component, pattern in components.items():
        if pattern in html_content:
            print(f"✅ {component} found")
        else:
            print(f"❌ {component} not found")
            all_found = False

    return all_found


def validate_javascript_functionality():
    """Validate JavaScript functionality"""
    print("\n=== Validating JavaScript Functionality ===\n")

    # Check JavaScript for required functionality
    js_path = Path(__file__).resolve().parent.parent / "app" / "static" / "script.js"
    with open(js_path) as f:
        js_content = f.read()

    functionality = {
        "WebSocket Connection": "WebSocket(",
        "Message Sending": "sendMessage()",
        "Message Receiving": "handleChatMessage(",
        "Streaming Response": "appendToCurrentMessage(",
        "Knowledge Graph Update": "updateKnowledgeGraph(",
        "System Monitoring": "updateMetrics(",
        "Settings Management": "saveSettings()",
        "Sidebar Toggle": "toggleSidebar()",
        "Error Handling": "showError(",
        "Session Management": "loadOrCreateSessionId()",
    }

    all_found = True
    for feature, pattern in functionality.items():
        if pattern in js_content:
            print(f"✅ {feature} found")
        else:
            print(f"❌ {feature} not found")
            all_found = False

    return all_found


def run_interactive_validation(port=8000):
    """Run interactive validation"""
    print("\n=== Starting Interactive Validation ===\n")
    print(f"Opening web interface at http://localhost:{port}")
    print("Please test the following features:")
    print("1. Send and receive messages")
    print("2. Observe streaming responses")
    print("3. Toggle the knowledge graph sidebar")
    print("4. Check system monitoring updates")
    print("5. Open and use the settings modal")
    print("6. Test responsive design by resizing the browser window")
    print("\nPress Ctrl+C when done to stop the server")

    # Open browser
    webbrowser.open(f"http://localhost:{port}")

    # Serve static files
    serve_static_files(port)


def main():
    parser = argparse.ArgumentParser(description="Milestone 5 Validation Tool")
    parser.add_argument("--interactive", action="store_true", help="Run interactive validation")
    parser.add_argument("--port", type=int, default=8000, help="Port for HTTP server")

    args = parser.parse_args()

    try:
        if args.interactive:
            run_interactive_validation(args.port)
        else:
            # Run automated tests
            tests_passed = run_automated_tests()

            # Validate responsive design
            responsive_design_valid = validate_responsive_design()

            # Validate UI components
            ui_components_valid = validate_ui_components()

            # Validate JavaScript functionality
            js_functionality_valid = validate_javascript_functionality()

            # Print summary
            print("\n=== Validation Summary ===\n")
            print(f"Automated Tests: {'✅ Passed' if tests_passed else '❌ Failed'}")
            print(f"Responsive Design: {'✅ Valid' if responsive_design_valid else '❌ Invalid'}")
            print(f"UI Components: {'✅ Valid' if ui_components_valid else '❌ Invalid'}")
            print(
                f"JavaScript Functionality: {'✅ Valid' if js_functionality_valid else '❌ Invalid'}"
            )

            if (
                tests_passed
                and responsive_design_valid
                and ui_components_valid
                and js_functionality_valid
            ):
                print("\n✅ All validation checks passed!")
                print(
                    "\nTo run interactive validation, use: python cli/validate_milestone_5.py --interactive"
                )
            else:
                print("\n❌ Some validation checks failed. Please fix the issues and try again.")
    except Exception as e:
        print(f"\n❌ Error during validation: {e}")
        import traceback

        traceback.print_exc()
        print("\nPlease try running with --interactive flag for a more robust test.")


if __name__ == "__main__":
    main()
