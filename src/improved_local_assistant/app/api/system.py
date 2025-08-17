"""
System API endpoints for the Improved Local AI Assistant.

This module provides REST API endpoints for system information and health checks.
"""

import asyncio
import logging
import time
from datetime import datetime

import psutil
from fastapi import APIRouter
from fastapi import HTTPException
from fastapi import Request
from fastapi.responses import JSONResponse

router = APIRouter()
logger = logging.getLogger(__name__)


@router.get("/health")
async def health_check(request: Request):
    """Health check endpoint."""
    start_time = time.time()

    model_manager = getattr(request.app.state, "model_manager", None)
    kg_manager = getattr(request.app.state, "kg_manager", None)
    conversation_manager = getattr(request.app.state, "conversation_manager", None)
    system_monitor = getattr(request.app.state, "system_monitor", None)
    connection_manager = getattr(request.app.state, "connection_manager", None)

    services_status = {
        "model_manager": model_manager is not None,
        "kg_manager": kg_manager is not None,
        "conversation_manager": conversation_manager is not None,
        "system_monitor": system_monitor is not None,
    }

    if not all([model_manager, conversation_manager]):
        return JSONResponse(
            status_code=503,
            content={
                "status": "error",
                "message": "Critical services not initialized",
                "services": services_status,
                "timestamp": datetime.now().isoformat(),
                "response_time_ms": int((time.time() - start_time) * 1000),
            },
        )

    try:
        response_time_ms = int((time.time() - start_time) * 1000)

        return {
            "status": "ok",
            "timestamp": datetime.now().isoformat(),
            "response_time_ms": response_time_ms,
            "services": services_status,
            "version": "1.0.0",
        }
    except Exception as e:
        logger.error(f"Health check failed: {str(e)}")
        response_time_ms = int((time.time() - start_time) * 1000)
        return JSONResponse(
            status_code=500,
            content={
                "status": "error",
                "message": str(e),
                "timestamp": datetime.now().isoformat(),
                "response_time_ms": response_time_ms,
            },
        )