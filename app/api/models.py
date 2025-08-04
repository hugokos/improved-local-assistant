"""
Model Management API endpoints.

This module provides REST API endpoints for:
• Switching conversation and knowledge extraction models
• Getting available model options
• Checking current model status
• Model availability testing
"""

from fastapi import APIRouter, HTTPException, Depends
from pydantic import BaseModel
from typing import Any, Dict, List, Optional

from app.core.dependencies import get_dynamic_model_manager
from services.dynamic_model_manager import DynamicModelManager


router = APIRouter(prefix="/api/models", tags=["models"])


class ModelSwitchRequest(BaseModel):
    """Request model for model switching."""
    model_name: str


class ModelSwitchResponse(BaseModel):
    """Response model for model switching."""
    success: bool
    message: str
    old_model: Optional[str] = None
    new_model: Optional[str] = None
    switch_time: Optional[float] = None
    error: Optional[str] = None
    config: Optional[Dict[str, Any]] = None


class ModelStatusResponse(BaseModel):
    """Response model for model status."""
    current_models: Dict[str, str]
    available_models: Dict[str, List[Dict[str, Any]]]
    metrics: Dict[str, Any]


@router.post("/conversation/switch", response_model=ModelSwitchResponse)
async def switch_conversation_model(
    request: ModelSwitchRequest,
    model_manager: DynamicModelManager = Depends(get_dynamic_model_manager)
):
    """
    Switch the conversation model to the specified model.
    
    Args:
        request: Model switch request containing the target model name
        model_manager: Dynamic model manager instance
        
    Returns:
        ModelSwitchResponse: Result of the model switch operation
    """
    try:
        result = await model_manager.switch_conversation_model(request.model_name)
        
        return ModelSwitchResponse(
            success=result["success"],
            message=result.get("message", ""),
            old_model=result.get("old_model"),
            new_model=result.get("new_model"),
            switch_time=result.get("switch_time"),
            error=result.get("error"),
            config=result.get("model_config")
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to switch conversation model: {str(e)}")


@router.post("/knowledge/switch", response_model=ModelSwitchResponse)
async def switch_knowledge_model(
    request: ModelSwitchRequest,
    model_manager: DynamicModelManager = Depends(get_dynamic_model_manager)
):
    """
    Switch the knowledge extraction model to the specified model.
    
    Args:
        request: Model switch request containing the target model name
        model_manager: Dynamic model manager instance
        
    Returns:
        ModelSwitchResponse: Result of the model switch operation
    """
    try:
        result = await model_manager.switch_knowledge_model(request.model_name)
        
        return ModelSwitchResponse(
            success=result["success"],
            message=result.get("message", ""),
            old_model=result.get("old_model"),
            new_model=result.get("new_model"),
            switch_time=result.get("switch_time"),
            error=result.get("error"),
            config=result.get("model_config")
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to switch knowledge model: {str(e)}")


@router.get("/status", response_model=ModelStatusResponse)
async def get_model_status(
    model_manager: DynamicModelManager = Depends(get_dynamic_model_manager)
):
    """
    Get current model status and available options.
    
    Args:
        model_manager: Dynamic model manager instance
        
    Returns:
        ModelStatusResponse: Current model status and available options
    """
    try:
        status = model_manager.get_status()
        
        return ModelStatusResponse(
            current_models=status["current_models"],
            available_models=status["available_models"],
            metrics=status["metrics"]
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get model status: {str(e)}")


@router.get("/available")
async def get_available_models(
    model_manager: DynamicModelManager = Depends(get_dynamic_model_manager)
):
    """
    Get available model options for conversation and knowledge extraction.
    
    Args:
        model_manager: Dynamic model manager instance
        
    Returns:
        Dict: Available model options
    """
    try:
        return model_manager.get_available_models()
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get available models: {str(e)}")


@router.get("/current")
async def get_current_models(
    model_manager: DynamicModelManager = Depends(get_dynamic_model_manager)
):
    """
    Get currently active models.
    
    Args:
        model_manager: Dynamic model manager instance
        
    Returns:
        Dict: Currently active models
    """
    try:
        return model_manager.get_current_models()
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get current models: {str(e)}")


@router.get("/metrics")
async def get_model_metrics(
    model_manager: DynamicModelManager = Depends(get_dynamic_model_manager)
):
    """
    Get model management metrics.
    
    Args:
        model_manager: Dynamic model manager instance
        
    Returns:
        Dict: Model management metrics
    """
    try:
        return model_manager.get_metrics()
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get model metrics: {str(e)}")