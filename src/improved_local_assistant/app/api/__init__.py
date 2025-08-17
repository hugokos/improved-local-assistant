"""API routers for the Improved Local AI Assistant."""

from improved_local_assistant.app.api.chat import router as chat_router
from improved_local_assistant.app.api.graph import router as graph_router
from improved_local_assistant.app.api.system import router as system_router
from fastapi import APIRouter

# Create a combined router
api_router = APIRouter()

# Include domain-specific routers
api_router.include_router(chat_router, prefix="/api", tags=["chat"])
api_router.include_router(graph_router, prefix="/api", tags=["graph"])
api_router.include_router(system_router, prefix="/api", tags=["system"])

__all__ = ["api_router"]
