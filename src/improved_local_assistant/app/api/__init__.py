"""API routers for the Improved Local AI Assistant."""

from fastapi import APIRouter

from app.api.chat import router as chat_router
from app.api.graph import router as graph_router
from app.api.system import router as system_router

# Create a combined router
api_router = APIRouter()

# Include domain-specific routers
api_router.include_router(chat_router, prefix="/api", tags=["chat"])
api_router.include_router(graph_router, prefix="/api", tags=["graph"])
api_router.include_router(system_router, prefix="/api", tags=["system"])

__all__ = ["api_router"]
