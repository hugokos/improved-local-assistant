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
        # Get timeout function from app state or import
        get_timeout = getattr(request.app.state, "get_timeout", None)
        if not get_timeout:
            from services import get_timeout

        # Get config from app state
        config = getattr(request.app.state, "config", {})

        # model status (timeout)
        try:
            model_status = await asyncio.wait_for(
                model_manager.get_model_status(),
                timeout=get_timeout("health_check_timeout", config),
            )
        except asyncio.TimeoutError:
            model_status = {"error": "Timeout checking model status"}
        except Exception as e:
            model_status = {"error": str(e)}

        # kg status
        kg_status = {"total_graphs": 0, "total_nodes": 0, "total_edges": 0}
        if kg_manager:
            try:
                kg_status = kg_manager.get_graph_statistics()
            except Exception as e:
                kg_status = {"error": str(e)}

        # system health
        system_health = {"status": "unknown", "components": {}}
        if system_monitor:
            try:
                system_health = system_monitor.check_health()
            except Exception as e:
                system_health = {"status": "error", "error": str(e)}

        active_connections_count = (
            len(connection_manager.active_connections) if connection_manager else 0
        )
        session_count = len(conversation_manager.sessions) if conversation_manager else 0

        overall_status = "ok"
        reasons = []
        if "error" in model_status:
            overall_status = "warning"
            reasons.append(f"Model error: {model_status['error']}")
        if system_health["status"] != "ok":
            overall_status = system_health["status"]
            reasons.append("System health issue")

        if system_monitor:
            usage = system_monitor.get_resource_usage()
            if usage["cpu_percent"] > 90:
                overall_status = "warning"
                reasons.append(f"High CPU usage: {usage['cpu_percent']}%")
            if usage["memory_percent"] > 90:
                overall_status = "warning"
                reasons.append(f"High memory usage: {usage['memory_percent']}%")

        response_time_ms = int((time.time() - start_time) * 1000)

        return {
            "status": overall_status,
            "status_reasons": reasons or None,
            "timestamp": datetime.now().isoformat(),
            "uptime": system_monitor.metrics["system"]["uptime_seconds"]
            if system_monitor
            else None,
            "response_time_ms": response_time_ms,
            "services": {
                "model_manager": "ok" if "error" not in model_status else "warning",
                "kg_manager": "ok" if "error" not in kg_status else "warning",
                "conversation_manager": "ok",
                "system_monitor": system_health["status"],
            },
            "models": {
                "conversation": model_manager.conversation_model if model_manager else None,
                "knowledge": model_manager.knowledge_model if model_manager else None,
                "status": model_status.get("models", {}),
            },
            "knowledge_graphs": {
                "count": kg_status.get("total_graphs", 0),
                "nodes": kg_status.get("total_nodes", 0),
                "edges": kg_status.get("total_edges", 0),
            },
            "system": {
                "components": system_health.get("components", {}),
                "resource_usage": system_monitor.get_resource_usage() if system_monitor else {},
                "active_connections": active_connections_count,
                "active_sessions": session_count,
            },
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


@router.get("/metrics")
async def get_metrics(request: Request):
    """Get system metrics."""
    try:
        system_monitor = getattr(request.app.state, "system_monitor", None)
        model_manager = getattr(request.app.state, "model_manager", None)
        kg_manager = getattr(request.app.state, "kg_manager", None)
        conversation_manager = getattr(request.app.state, "conversation_manager", None)

        if system_monitor:
            resource_usage = system_monitor.get_resource_usage()
            performance_metrics = system_monitor.get_performance_metrics()
        else:
            cpu_percent = psutil.cpu_percent(interval=0.1)
            memory = psutil.virtual_memory()
            resource_usage = {
                "cpu_percent": cpu_percent,
                "memory_percent": memory.percent,
                "memory_used_gb": round(memory.used / (1024**3), 2),
                "memory_total_gb": round(memory.total / (1024**3), 2),
            }
            performance_metrics = {}

        model_metrics = model_manager.metrics if model_manager else {}
        kg_metrics = kg_manager.metrics if kg_manager else {}
        conversation_metrics = conversation_manager.metrics if conversation_manager else {}

        return {
            "timestamp": datetime.now().isoformat(),
            "resource_usage": resource_usage,
            "performance_metrics": performance_metrics,
            "model_metrics": model_metrics,
            "kg_metrics": kg_metrics,
            "conversation_metrics": conversation_metrics,
        }
    except Exception as e:
        logger.error(f"Error getting metrics: {str(e)}")
        return JSONResponse(status_code=500, content={"status": "error", "message": str(e)})


@router.get("/system/info")
async def get_system_info(request: Request):
    """Get detailed system information."""
    try:
        system_monitor = getattr(request.app.state, "system_monitor", None)
        if not system_monitor:
            raise HTTPException(status_code=503, detail="System monitor not initialized")

        return system_monitor.get_all_metrics()
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting system info: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))
