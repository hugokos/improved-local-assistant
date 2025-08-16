"""
FastAPI application factory for Improved Local Assistant.
"""
import asyncio
import logging
import os
import sys
from pathlib import Path
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

# Apply UTF-8 runtime patch for Windows compatibility
try:
    from ..services.utf8_runtime_patch import apply_utf8_patch
    apply_utf8_patch()
except ImportError:
    pass  # Service may not exist in new structure

# Fix for Windows asyncio event loop issue
if sys.platform == "win32":
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

from ..core.settings import get_settings


logger = logging.getLogger(__name__)


def setup_logging(settings) -> None:
    """Setup logging configuration."""
    log_dir = Path("logs")
    log_dir.mkdir(exist_ok=True)
    
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler(log_dir / "app.log", mode="a"),
        ],
    )
    
    # Reduce noise from verbose libraries
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("httpcore").setLevel(logging.WARNING)
    logging.getLogger("urllib3").setLevel(logging.WARNING)
    logging.getLogger("sentence_transformers").setLevel(logging.WARNING)


def create_directories(settings) -> None:
    """Create necessary directories."""
    directories = [
        settings.data_dir,
        settings.prebuilt_dir,
        settings.data_dir / "dynamic_graph",
        settings.data_dir / "sessions",
        Path("logs"),
    ]
    
    for directory in directories:
        Path(directory).mkdir(parents=True, exist_ok=True)


def include_routers(app: FastAPI) -> None:
    """Include API routers."""
    try:
        from .routes import chat, graph, models, system
        
        app.include_router(chat.router, prefix="/api", tags=["chat"])
        app.include_router(graph.router, prefix="/api", tags=["graph"])
        app.include_router(models.router, prefix="/api", tags=["models"])
        app.include_router(system.router, prefix="/api", tags=["system"])
        
    except ImportError as e:
        logger.warning(f"Could not import some routes: {e}")


def init_services(app: FastAPI, settings) -> None:
    """Initialize application services."""
    # This will be implemented as services are refactored
    pass


def create_app() -> FastAPI:
    """Create and configure FastAPI application."""
    settings = get_settings()
    
    # Setup logging and directories
    setup_logging(settings)
    create_directories(settings)
    
    # Create FastAPI app
    app = FastAPI(
        title="Improved Local Assistant",
        description="Local-first GraphRAG assistant with offline voice interface",
        version="1.0.0",
        docs_url="/docs",
        redoc_url="/redoc",
    )
    
    # Add CORS middleware
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],  # Configure appropriately for production
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    
    # Mount static files if they exist
    static_dir = Path("src/improved_local_assistant/app/static")
    if static_dir.exists():
        app.mount("/static", StaticFiles(directory=str(static_dir)), name="static")
    
    # Include routers
    include_routers(app)
    
    # Initialize services
    init_services(app, settings)
    
    logger.info("FastAPI application created successfully")
    return app


# Factory function for uvicorn
def fastapi_app():
    """Factory function for uvicorn."""
    return create_app()
