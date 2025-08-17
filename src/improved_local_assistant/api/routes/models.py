"""
Model Management API endpoints.

This module provides REST API endpoints for:
• Getting available model options
• Checking current model status
• Model availability testing
"""

from typing import Any

from fastapi import APIRouter
from fastapi import HTTPException
from pydantic import BaseModel

router = APIRouter(prefix="/api/models", tags=["models"])


class ModelStatusResponse(BaseModel):
    """Response model for model status."""

    current_models: dict[str, str]
    available_models: dict[str, list[dict[str, Any]]]
    metrics: dict[str, Any]


@router.get("/status", response_model=ModelStatusResponse)
async def get_model_status():
    """
    Get current model status and available models.

    Returns:
        ModelStatusResponse: Current model status and available options
    """
    try:
        # Placeholder response - will be implemented when model management is ready
        return ModelStatusResponse(
            current_models={
                "conversation": "hermes3:3b",
                "knowledge": "phi3:mini"
            },
            available_models={
                "conversation": [
                    {"name": "hermes3:3b", "description": "Default conversation model"},
                    {"name": "llama3.2:3b", "description": "Alternative conversation model"}
                ],
                "knowledge": [
                    {"name": "phi3:mini", "description": "Default knowledge extraction model"},
                    {"name": "llama3.2:1b", "description": "Alternative knowledge model"}
                ]
            },
            metrics={
                "total_requests": 0,
                "successful_switches": 0,
                "last_switch_time": None
            }
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get model status: {str(e)}")


@router.get("/available")
async def get_available_models():
    """
    Get list of available models.

    Returns:
        dict: Available model options
    """
    try:
        return {
            "conversation_models": [
                {"name": "hermes3:3b", "description": "Default conversation model"},
                {"name": "llama3.2:3b", "description": "Alternative conversation model"}
            ],
            "knowledge_models": [
                {"name": "phi3:mini", "description": "Default knowledge extraction model"},
                {"name": "llama3.2:1b", "description": "Alternative knowledge model"}
            ]
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get available models: {str(e)}")
