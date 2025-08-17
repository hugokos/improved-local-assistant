"""
Model Management API endpoints.

This module provides REST API endpoints for model management.
"""

from typing import Any, Optional

from fastapi import APIRouter
from fastapi import HTTPException
from fastapi import Request
from pydantic import BaseModel

router = APIRouter()


class ModelSwitchRequest(BaseModel):
    """Request model for model switching."""

    model_name: str


class ModelSwitchResponse(BaseModel):
    """Response model for model switching."""

    success: bool
    message: str
    old_model: Optional[str] = None
    new_model: Optional[str] = None


@router.get("/status")
async def get_model_status(request: Request):
    """Get current model status."""
    try:
        model_manager = getattr(request.app.state, "model_manager", None)
        if not model_manager:
            return {"status": "Model manager not initialized"}
        
        return {"status": "ok", "models": "available"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/available")
async def get_available_models(request: Request):
    """Get available model options."""
    try:
        return {"conversation_models": [], "knowledge_models": []}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))