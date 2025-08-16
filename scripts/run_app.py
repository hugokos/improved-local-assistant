"""
Simple wrapper script to run the main application.

This script provides a simple way to run the main application
for the Improved Local AI Assistant.
"""

import argparse
import os
import shutil
import subprocess
import sys

# Add the project root to the Python path
project_root = os.path.dirname(os.path.abspath(__file__))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# Apply UTF-8 runtime patch for Windows compatibility
try:
    from services.utf8_runtime_patch import apply_utf8_patch

    apply_utf8_patch()
except ImportError:
    pass  # Patch not available, continue anyway


def main():
    """Run the main application."""
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description="Run the Improved Local AI Assistant")
    parser.add_argument(
        "--skip-survivalist", action="store_true", help="Skip loading the survivalist graph"
    )
    parser.add_argument(
        "--rebuild-survivalist",
        action="store_true",
        help="Rebuild the survivalist graph from scratch",
    )
    parser.add_argument("--port", type=int, default=8000, help="Port to run the application on")
    args = parser.parse_args()

    print("Starting Improved Local AI Assistant...")

    # Create necessary directories
    os.makedirs("logs", exist_ok=True)
    os.makedirs("data", exist_ok=True)
    os.makedirs("data/prebuilt_graphs", exist_ok=True)
    os.makedirs("data/dynamic_graph", exist_ok=True)
    os.makedirs("data/sessions", exist_ok=True)

    # Handle survivalist graph options
    survivalist_dir = os.path.join("data", "prebuilt_graphs", "survivalist")
    survivalist_disabled_dir = f"{survivalist_dir}_disabled"

    if args.skip_survivalist:
        print("Skipping survivalist graph as requested")
        if os.path.exists(survivalist_dir):
            print(f"Temporarily renaming {survivalist_dir} to {survivalist_disabled_dir}")
            if os.path.exists(survivalist_disabled_dir):
                shutil.rmtree(survivalist_disabled_dir)
            try:
                os.rename(survivalist_dir, survivalist_disabled_dir)
            except Exception as e:
                print(f"Warning: Could not rename directory: {e}")
    else:
        # Restore survivalist graph if it was disabled
        if not os.path.exists(survivalist_dir) and os.path.exists(survivalist_disabled_dir):
            print(f"Restoring survivalist graph from {survivalist_disabled_dir}")
            try:
                os.rename(survivalist_disabled_dir, survivalist_dir)
                print("Survivalist graph restored successfully")
            except Exception as e:
                print(f"Warning: Could not restore survivalist graph: {e}")
                print("Continuing anyway...")

    if args.rebuild_survivalist:
        print("Rebuilding survivalist graph as requested")
        try:
            # Run the rebuild script
            rebuild_script = os.path.join("scripts", "rebuild_survivalist_graph.py")
            subprocess.run([sys.executable, rebuild_script], check=True)
            print("Survivalist graph rebuilt successfully")
        except subprocess.CalledProcessError as e:
            print(f"Error rebuilding survivalist graph: {e}")
            print("Continuing without rebuilding...")

    # Set environment variable to control graph loading
    if args.skip_survivalist:
        os.environ["SKIP_SURVIVALIST_GRAPH"] = "1"

    # Run the main application with uvicorn
    try:
        # Try to run with uvicorn first (recommended for FastAPI)
        print(f"Starting server on port {args.port}...")
        subprocess.run(
            [
                sys.executable,
                "-m",
                "uvicorn",
                "app.main:app",
                "--host",
                "0.0.0.0",
                "--port",
                str(args.port),
            ],
            check=True,
        )
        return 0
    except subprocess.CalledProcessError as e:
        print(f"Error running with uvicorn: {e}")
        print("Trying to run directly...")

        # Fallback: run as module
        try:
            subprocess.run([sys.executable, "-m", "app.main"], check=True)
            return 0
        except subprocess.CalledProcessError as e2:
            print(f"Error running application: {e2}")
            return 1
    except KeyboardInterrupt:
        print("\nApplication stopped by user")
        return 0
    except Exception as e:
        print(f"Unexpected error: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
