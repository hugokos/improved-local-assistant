"""
Graph API endpoints for the Improved Local AI Assistant.

This module provides REST API endpoints for knowledge graph operations.
"""

import asyncio
import logging
import os
from datetime import datetime
from typing import Optional

from improved_local_assistant.app.schemas import GraphCreateRequest
from improved_local_assistant.app.schemas import GraphRequest
from improved_local_assistant.app.schemas import GraphTraversalRequest
from fastapi import APIRouter
from fastapi import File
from fastapi import Form
from fastapi import HTTPException
from fastapi import Query
from fastapi import Request
from fastapi import UploadFile
from fastapi.responses import HTMLResponse
from fastapi.responses import StreamingResponse
from improved_local_assistant.services import HTTP_STATUS
from starlette.background import BackgroundTask

router = APIRouter()
logger = logging.getLogger(__name__)


@router.get("/graphs")
async def list_graphs(request: Request):
    """List all available knowledge graphs."""
    try:
        kg_manager = request.app.state.kg_manager
        if not kg_manager:
            raise HTTPException(status_code=503, detail="Knowledge graph manager not initialized")

        stats = kg_manager.get_graph_statistics()
        graphs = [
            {
                "id": gid,
                "nodes": g.get("nodes", 0),
                "edges": g.get("edges", 0),
                "density": g.get("density", 0),
                "error": g.get("error"),
            }
            for gid, g in stats["graphs"].items()
        ]
        return {
            "graphs": graphs,
            "total_graphs": stats["total_graphs"],
            "total_nodes": stats["total_nodes"],
            "total_edges": stats["total_edges"],
        }
    except Exception as e:
        logger.error(f"Error listing graphs: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/graph/query")
async def query_graph(request: Request, graph_request: GraphRequest):
    """Query knowledge graphs."""
    try:
        kg_manager = request.app.state.kg_manager
        if not kg_manager:
            raise HTTPException(status_code=503, detail="Knowledge graph manager not initialized")

        result = kg_manager.query_graphs(graph_request.query)
        return {
            "response": result["response"],
            "source_nodes": result.get("source_nodes", []),
            "metadata": result["metadata"],
        }
    except Exception as e:
        logger.error(f"Error in graph query endpoint: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/graph/visualize")
async def visualize_graph(request: Request, graph_id: Optional[str] = None):
    """Generate HTML visualization of knowledge graph."""
    try:
        kg_manager = request.app.state.kg_manager
        if not kg_manager:
            raise HTTPException(status_code=503, detail="Knowledge graph manager not initialized")

        # Run visualization in thread to avoid blocking
        html_content = await asyncio.to_thread(kg_manager.visualize_graph, graph_id)
        return HTMLResponse(content=html_content)
    except Exception as e:
        logger.error(f"Error in graph visualization endpoint: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))