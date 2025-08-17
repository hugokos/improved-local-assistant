"""
FastAPI application with WebSocket support for the Improved Local AI Assistant.

This module provides the main FastAPI application with WebSocket endpoints for
real-time chat, session management, and proper error handling.
"""

import asyncio
import os
import sys

# Fix for Windows asyncio event loop issue
if sys.platform == "win32":
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

# Add the project root to the Python path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from improved_local_assistant.app.api import api_router  # noqa: E402
from improved_local_assistant.app.api.models import router as models_router  # noqa: E402
from improved_local_assistant.app.core import ConnectionManager  # noqa: E402
from improved_local_assistant.app.core import load_config  # noqa: E402
from improved_local_assistant.app.core import setup_logging  # noqa: E402
from improved_local_assistant.app.services.init import init_app  # noqa: E402
from improved_local_assistant.app.ws.chat import chat_websocket  # noqa: E402
from improved_local_assistant.app.ws.monitor import monitor_websocket  # noqa: E402
from fastapi import FastAPI  # noqa: E402
from fastapi import WebSocket  # noqa: E402
from fastapi.middleware.cors import CORSMiddleware  # noqa: E402
from fastapi.responses import FileResponse  # noqa: E402
from fastapi.responses import HTMLResponse  # noqa: E402
from fastapi.staticfiles import StaticFiles  # noqa: E402
from starlette.middleware.trustedhost import TrustedHostMiddleware  # noqa: E402

# Setup logging
logger = setup_logging()

# Load configuration
config = load_config()

# Create FastAPI application
app = FastAPI(
    title="Improved Local AI Assistant",
    description="A local AI assistant with dual-model architecture and knowledge graph integration",
    version="1.0.0",
)

# Configure CORS middleware with default values (will be updated in startup if needed)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Default - will be updated from config
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Add TrustedHostMiddleware to allow localhost connections
app.add_middleware(
    TrustedHostMiddleware, 
    allowed_hosts=["localhost", "127.0.0.1", "[::1]", "*.localhost"]
)

# Serve static files
from pathlib import Path
BASE_DIR = Path(__file__).resolve().parent
STATIC_DIR = (BASE_DIR / "static").resolve()

app.mount(
    "/static",
    StaticFiles(directory=str(STATIC_DIR)),
    name="static",
)

# Create connection manager
connection_manager = ConnectionManager()
app.state.connection_manager = connection_manager
app.state.config = config

# Include API routers
app.include_router(api_router)
app.include_router(models_router)


# Root endpoint
@app.get("/", response_class=HTMLResponse)
async def get_root():
    """Serve the main HTML page."""
    # Use FileResponse to avoid blocking read in event loop
    index_path = STATIC_DIR / "index.html"
    if not index_path.exists():
        from fastapi import HTTPException

        raise HTTPException(status_code=404, detail="index.html not found")
    return FileResponse(str(index_path))


# Demo endpoint
@app.get("/demo.html", response_class=HTMLResponse)
async def get_demo():
    """Serve the demo HTML page."""
    demo_path = STATIC_DIR / "demo.html"
    if not demo_path.exists():
        from fastapi import HTTPException

        raise HTTPException(status_code=404, detail="demo.html not found")
    return FileResponse(str(demo_path))


# Favicon endpoint to prevent 404 errors
@app.get("/favicon.ico")
async def get_favicon():
    """Return empty response for favicon to prevent 404 logs."""
    from fastapi import Response

    return Response(status_code=204)


# WebSocket endpoints
@app.websocket("/ws/{session_id}")
async def websocket_endpoint(websocket: WebSocket, session_id: str):
    """WebSocket endpoint for real-time chat."""
    await chat_websocket(websocket, session_id, app)


@app.websocket("/ws/monitor")
async def websocket_monitor(websocket: WebSocket):
    """WebSocket endpoint for real-time system monitoring."""
    await monitor_websocket(websocket, app)


# Voice WebSocket endpoints
@app.websocket("/ws/stt/{session_id}")
async def websocket_stt(websocket: WebSocket, session_id: str):
    """WebSocket endpoint for speech-to-text processing."""
    from improved_local_assistant.app.ws.voice_stt import stt_websocket

    await stt_websocket(websocket, session_id, app)


@app.websocket("/ws/tts/{session_id}")
async def websocket_tts(websocket: WebSocket, session_id: str):
    """WebSocket endpoint for text-to-speech processing."""
    from improved_local_assistant.app.ws.voice_tts import tts_websocket

    await tts_websocket(websocket, session_id, app)


# Initialize services
init_app(app, config)


@app.on_event("startup")
async def log_routes():
    """Log all registered routes on startup for debugging."""
    logger.info("=== Registered Routes ===")
    for route in app.router.routes:
        try:
            path = getattr(route, "path", getattr(route, "path_format", ""))
            methods = getattr(route, "methods", None) or []
            name = getattr(route, "name", "")
            route_type = "WebSocket" if hasattr(route, "endpoint") and "websocket" in str(route.endpoint).lower() else "HTTP"
            logger.info(f"Route: {route_type} {methods or ['WS']} {path} ({name})")
        except Exception as e:
            logger.warning(f"Could not log route: {e}")
    logger.info("=== End Routes ===")

# Run the application
if __name__ == "__main__":
    import uvicorn

    api_config = config.get("api", {})
    host = api_config.get("host", "localhost")
    port = api_config.get("port", 8000)

    # Run the app directly when executed as main
    uvicorn.run(app, host=host, port=port, reload=False)
