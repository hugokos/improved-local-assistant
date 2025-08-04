#!/usr/bin/env python3
"""
Production-ready launcher for the Improved Local AI Assistant.

This script provides optimized startup with configurable performance settings,
lazy loading, and proper error handling for production deployment.
"""

import argparse
import asyncio
import logging
import os
import platform
import shutil
import sys
import time
from pathlib import Path

# Apply UTF-8 runtime patch for Windows compatibility
from services.utf8_runtime_patch import apply_utf8_patch
apply_utf8_patch()

# 🔧 VERIFICATION: Check if UTF-8 patch is active (early verification)
import io
b = io.BytesIO(b'\xc3\xa9')  # 'é' in UTF-8
try:
    b.read().decode(sys.getdefaultencoding())
    print(f"[DEBUG] Default encoding is UTF-8: {sys.getdefaultencoding()}")
except UnicodeDecodeError:
    print(f"[DEBUG] Default encoding is NOT UTF-8: {sys.getdefaultencoding()}")

# Fix for Windows asyncio event loop issue
if sys.platform == "win32":
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

# Add the project root to the Python path
project_root = Path(__file__).parent.absolute()
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))


def setup_logging(level: str = "INFO") -> None:
    """Setup logging with appropriate level."""
    logging.basicConfig(
        level=getattr(logging, level.upper()),
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler("logs/app.log", mode="a"),
        ]
    )
    
    # Reduce noise from verbose libraries during startup
    if level.upper() != "DEBUG":
        logging.getLogger("httpx").setLevel(logging.WARNING)
        logging.getLogger("httpcore").setLevel(logging.WARNING)
        logging.getLogger("urllib3").setLevel(logging.WARNING)
        logging.getLogger("sentence_transformers").setLevel(logging.WARNING)


def create_directories() -> None:
    """Create necessary directories."""
    directories = [
        "logs",
        "data",
        "data/prebuilt_graphs", 
        "data/dynamic_graph",
        "data/sessions",
    ]
    
    for directory in directories:
        os.makedirs(directory, exist_ok=True)


def handle_survivalist_graph(args) -> None:
    """Handle survivalist graph enable/disable."""
    survivalist_dir = Path("data/prebuilt_graphs/survivalist")
    survivalist_disabled_dir = Path("data/prebuilt_graphs/survivalist_disabled")
    
    if args.skip_survivalist:
        if survivalist_dir.exists():
            print(f"Disabling survivalist graph: {survivalist_dir} -> {survivalist_disabled_dir}")
            if survivalist_disabled_dir.exists():
                shutil.rmtree(survivalist_disabled_dir)
            survivalist_dir.rename(survivalist_disabled_dir)
            os.environ["SKIP_SURVIVALIST_GRAPH"] = "1"
    else:
        # Restore survivalist graph if it was disabled
        if not survivalist_dir.exists() and survivalist_disabled_dir.exists():
            print(f"Restoring survivalist graph: {survivalist_disabled_dir} -> {survivalist_dir}")
            survivalist_disabled_dir.rename(survivalist_dir)
            print("Survivalist graph restored successfully")


def set_performance_environment(args) -> None:
    """Set environment variables for performance optimization."""
    # Startup mode
    os.environ["STARTUP_PRELOAD"] = args.preload
    os.environ["GRAPH_LAZY_LOAD"] = str(args.lazy_load_graphs).lower()
    os.environ["OLLAMA_HEALTHCHECK"] = args.ollama_healthcheck
    
    # Embedding model settings
    os.environ["EMBED_MODEL_DEVICE"] = args.embed_device
    os.environ["EMBED_MODEL_INT8"] = str(args.embed_int8).lower()
    os.environ["EMBED_MODEL_PATH"] = "BAAI/bge-small-en-v1.5"  # Ensure correct model
    
    # Resource monitoring
    os.environ["RESOURCE_MONITOR_INTERVAL"] = str(args.monitor_interval)
    os.environ["RESOURCE_MONITOR_DEBOUNCE"] = str(args.monitor_debounce)
    
    # Memory thresholds
    os.environ["MEM_PRESSURE_THRESHOLD"] = str(args.memory_threshold)
    
    # Platform-specific optimizations
    if platform.system() == "Windows":
        os.environ["SKIP_PROCESS_PRIORITY"] = "1"
        # Set UTF-8 encoding for Windows
        os.environ["PYTHONIOENCODING"] = "utf-8"


def main():
    """Main application entry point."""
    parser = argparse.ArgumentParser(
        description="Improved Local AI Assistant - Production Ready",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # Basic options
    parser.add_argument("--port", type=int, default=8000, help="Port to run the application on")
    parser.add_argument("--host", default="0.0.0.0", help="Host to bind to")
    parser.add_argument("--log-level", default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR"])
    
    # Performance options
    parser.add_argument("--preload", default="models", choices=["none", "models", "graphs", "all"],
                       help="What to preload during startup")
    parser.add_argument("--lazy-load-graphs", action="store_true", default=True,
                       help="Load graphs in background after startup")
    parser.add_argument("--ollama-healthcheck", default="version", choices=["version", "chat"],
                       help="Type of Ollama health check to perform")
    
    # Embedding model options
    parser.add_argument("--embed-device", default="cpu", choices=["cpu", "cuda"],
                       help="Device for embedding model")
    parser.add_argument("--embed-int8", action="store_true", default=True,
                       help="Use int8 quantization for embedding model")
    
    # Resource monitoring options
    parser.add_argument("--monitor-interval", type=int, default=10,
                       help="Resource monitoring interval in seconds")
    parser.add_argument("--monitor-debounce", type=int, default=60,
                       help="Debounce time for resource cleanup actions")
    parser.add_argument("--memory-threshold", type=float, default=0.95,
                       help="Memory pressure threshold (0.0-1.0)")
    
    # Graph options
    parser.add_argument("--skip-survivalist", action="store_true",
                       help="Skip loading the survivalist graph")
    parser.add_argument("--rebuild-survivalist", action="store_true",
                       help="Rebuild the survivalist graph from scratch")
    
    # Development options
    parser.add_argument("--reload", action="store_true", help="Enable auto-reload for development")
    
    args = parser.parse_args()
    
    print("🚀 Starting Improved Local AI Assistant (Production Mode)")
    print(f"   Preload mode: {args.preload}")
    print(f"   Lazy load graphs: {args.lazy_load_graphs}")
    print(f"   Ollama health check: {args.ollama_healthcheck}")
    print(f"   Memory threshold: {args.memory_threshold}")
    
    # Setup
    setup_logging(args.log_level)
    create_directories()
    handle_survivalist_graph(args)
    set_performance_environment(args)
    
    # Handle graph rebuilding
    if args.rebuild_survivalist:
        print("Rebuilding survivalist graph...")
        try:
            import subprocess
            rebuild_script = Path("scripts/rebuild_survivalist_graph.py")
            if rebuild_script.exists():
                subprocess.run([sys.executable, str(rebuild_script)], check=True)
                print("✅ Survivalist graph rebuilt successfully")
            else:
                print("⚠️  Rebuild script not found, skipping")
        except subprocess.CalledProcessError as e:
            print(f"❌ Error rebuilding survivalist graph: {e}")
            print("Continuing without rebuilding...")
    
    # Start the application
    try:
        import uvicorn
        
        print(f"🌐 Starting server on {args.host}:{args.port}")
        print("📊 Performance optimizations enabled")
        print("🔄 Use Ctrl+C to stop")
        
        uvicorn.run(
            "app.main:app",
            host=args.host,
            port=args.port,
            reload=args.reload,
            log_level=args.log_level.lower(),
            access_log=args.log_level.upper() == "DEBUG",
        )
        
    except KeyboardInterrupt:
        print("\n🛑 Application stopped by user")
        return 0
    except Exception as e:
        print(f"❌ Failed to start application: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())