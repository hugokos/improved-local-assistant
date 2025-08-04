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

from fastapi import FastAPI, WebSocket
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles

from app.api import api_router
from app.api.models import router as models_router
from app.core import setup_logging, load_config, ConnectionManager
from app.services.init import init_app
from app.ws.chat import chat_websocket
from app.ws.monitor import monitor_websocket

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

# Serve static files
app.mount("/static", StaticFiles(directory=os.path.join(os.path.dirname(__file__), "static")), name="static")

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
    path = "app/static/index.html"
    if not os.path.exists(path):
        from fastapi import HTTPException
        raise HTTPException(status_code=404, detail="index.html not found")
    return FileResponse(path)

# Demo endpoint
@app.get("/demo.html", response_class=HTMLResponse)
async def get_demo():
    """Serve the demo HTML page."""
    path = "demo.html"
    if not os.path.exists(path):
        from fastapi import HTTPException
        raise HTTPException(status_code=404, detail="demo.html not found")
    return FileResponse(path)

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

# Initialize services
init_app(app, config)

# Run the application
if __name__ == "__main__":
    import uvicorn

    api_config = config.get("api", {})
    host = api_config.get("host", "localhost")
    port = api_config.get("port", 8000)

    # Run the app directly when executed as main
    uvicorn.run(app, host=host, port=port, reload=False)