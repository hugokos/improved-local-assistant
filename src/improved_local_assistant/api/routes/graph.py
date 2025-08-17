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


@router.post("/graph/create")
async def create_graph(request: Request, graph_request: GraphCreateRequest):
    """Create a new knowledge graph from documents."""
    try:
        kg_manager = request.app.state.kg_manager
        connection_manager = request.app.state.connection_manager

        if not kg_manager:
            raise HTTPException(status_code=503, detail="Knowledge graph manager not initialized")

        graph_id = kg_manager.create_graph_from_documents(
            graph_request.docs_path, graph_request.graph_id
        )
        if not graph_id:
            raise HTTPException(status_code=500, detail="Failed to create graph")

        if connection_manager:
            await connection_manager.notify_graph_update(graph_id, "created")

        return {"status": "success", "graph_id": graph_id}
    except Exception as e:
        logger.error(f"Error creating graph: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/graph/traversal")
async def graph_traversal(request: Request, traversal_request: GraphTraversalRequest):
    """Perform graph traversal between source and target nodes."""
    try:
        kg_manager = request.app.state.kg_manager
        if not kg_manager:
            raise HTTPException(status_code=503, detail="Knowledge graph manager not initialized")

        paths = kg_manager.get_graph_traversal(
            traversal_request.source, traversal_request.target, traversal_request.max_hops
        )

        return {
            "paths": paths,
            "count": len(paths),
            "source": traversal_request.source,
            "target": traversal_request.target,
            "max_hops": traversal_request.max_hops,
        }
    except Exception as e:
        logger.error(f"Error in graph traversal: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/graph/load")
async def load_graph(request: Request, path: str):
    """Load a pre-built knowledge graph from a directory."""
    try:
        kg_manager = request.app.state.kg_manager
        connection_manager = request.app.state.connection_manager

        if not kg_manager:
            raise HTTPException(status_code=503, detail="Knowledge graph manager not initialized")

        graph_id = kg_manager.add_new_graph(path)
        if not graph_id:
            raise HTTPException(status_code=500, detail="Failed to load graph")

        if connection_manager:
            await connection_manager.notify_graph_update(graph_id, "loaded")

        return {"status": "success", "graph_id": graph_id}
    except Exception as e:
        logger.error(f"Error loading graph: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/graph/{graph_id}/stats")
async def get_graph_stats(request: Request, graph_id: str):
    """Get statistics for a specific knowledge graph."""
    try:
        kg_manager = request.app.state.kg_manager
        if not kg_manager:
            raise HTTPException(status_code=503, detail="Knowledge graph manager not initialized")

        stats = kg_manager.get_graph_statistics()
        if graph_id not in stats["graphs"] and graph_id != "dynamic":
            raise HTTPException(status_code=404, detail=f"Graph {graph_id} not found")

        g = stats["graphs"].get(graph_id, {})
        return {
            "id": graph_id,
            "nodes": g.get("nodes", 0),
            "edges": g.get("edges", 0),
            "density": g.get("density", 0),
            "error": g.get("error"),
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting graph stats: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/graph/{graph_id}/export")
async def export_graph_interop(request: Request, graph_id: str, format: str = "networkx"):
    """Export a knowledge graph in interop format (optional - NetworkX/GraphML/etc)."""
    try:
        kg_manager = request.app.state.kg_manager
        if not kg_manager:
            raise HTTPException(
                status_code=HTTP_STATUS["SERVICE_UNAVAILABLE"],
                detail="Knowledge graph manager not initialized",
            )

        # For now, return basic stats - full interop export can be implemented later if needed
        stats = kg_manager.get_graph_statistics()
        if graph_id not in stats["graphs"] and graph_id != "dynamic":
            raise HTTPException(status_code=404, detail=f"Graph {graph_id} not found")

        if format == "networkx":
            return {
                "graph_id": graph_id,
                "format": format,
                "nodes": stats["graphs"].get(graph_id, {}).get("nodes", 0),
                "edges": stats["graphs"].get(graph_id, {}).get("edges", 0),
                "timestamp": datetime.now().isoformat(),
                "note": "Full NetworkX export not implemented - use /export_native for complete graph data",
            }
        else:
            raise HTTPException(
                status_code=400,
                detail=f"Unsupported export format: {format}. Use /export_native for complete graph export.",
            )
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error exporting graph: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/graph/{graph_id}/export_native")
async def export_graph_native(
    request: Request,
    graph_id: str,
    limit: Optional[int] = Query(None, description="Limit number of nodes in export"),
    hops: Optional[int] = Query(None, description="Maximum hops from seed nodes"),
    max_nodes: Optional[int] = Query(None, description="Maximum nodes in partial export"),
):
    """Export a knowledge graph in native LlamaIndex persistence format."""
    try:
        kg_manager = request.app.state.kg_manager
        if not kg_manager:
            raise HTTPException(
                status_code=HTTP_STATUS["SERVICE_UNAVAILABLE"],
                detail="Knowledge graph manager not initialized",
            )

        # Export graph (run in thread to avoid blocking)
        # If partial export parameters are provided, create a subgraph first
        if any([limit, hops, max_nodes]):
            # For partial exports, we need to extract a subgraph first
            # This is a simplified approach - in production you might want more sophisticated seed node selection
            subgraph_params = {"max_hops": hops or 2, "max_nodes": max_nodes or limit or 100}
            # Use a generic query for now - could be enhanced to accept seed nodes
            zip_path = await asyncio.to_thread(
                kg_manager.export_native_partial, graph_id, query="", **subgraph_params
            )
        else:
            # Full export
            zip_path = await asyncio.to_thread(kg_manager.export_native, graph_id)

        # Stream the zip file
        def cleanup_file():
            try:
                os.unlink(zip_path)
            except (OSError, FileNotFoundError):
                # File cleanup failed, but this is not critical
                pass
            except Exception as e:
                logger.warning(f"Failed to cleanup temporary zip file: {str(e)}")

        return StreamingResponse(
            open(zip_path, "rb"),
            media_type="application/zip",
            headers={"Content-Disposition": f"attachment; filename={graph_id}_export.zip"},
            background=BackgroundTask(cleanup_file),
        )

    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        logger.error(f"Error exporting graph: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/graph/import_native")
async def import_graph_native(
    request: Request,
    file: UploadFile = File(...),
    graph_id: str = Form(...),
    graph_type: str = Form("modular"),
    merge_strategy: str = Form("union"),
    replace: bool = Form(True, description="Replace existing graph (modular only)"),
):
    """Import a knowledge graph from native LlamaIndex persistence format."""
    try:
        kg_manager = request.app.state.kg_manager
        connection_manager = request.app.state.connection_manager

        if not kg_manager:
            raise HTTPException(
                status_code=HTTP_STATUS["SERVICE_UNAVAILABLE"],
                detail="Knowledge graph manager not initialized",
            )

        # Validate parameters
        if graph_type not in ["modular", "dynamic"]:
            raise HTTPException(status_code=400, detail="graph_type must be 'modular' or 'dynamic'")

        if merge_strategy not in ["union", "prefer_base", "prefer_incoming"]:
            raise HTTPException(
                status_code=400,
                detail="merge_strategy must be 'union', 'prefer_base', or 'prefer_incoming'",
            )

        # Validate file type
        if not file.filename.endswith(".zip"):
            raise HTTPException(status_code=400, detail="File must be a zip archive")

        # Save uploaded file temporarily
        import tempfile

        temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".zip")
        try:
            content = await file.read()
            temp_file.write(content)
            temp_file.close()

            # Validate file size (max 100MB for native imports)
            max_size = 100 * 1024 * 1024  # 100MB
            if len(content) > max_size:
                raise HTTPException(status_code=413, detail="File too large (max 100MB)")

            # Import graph (run in thread to avoid blocking)
            result_graph_id = await asyncio.to_thread(
                kg_manager.import_native,
                temp_file.name,
                graph_id,
                graph_type,
                merge_strategy,
                replace,
            )

            # Notify connected clients
            if connection_manager:
                await connection_manager.notify_graph_update(
                    result_graph_id, f"imported_{graph_type}"
                )

            return {
                "status": "success",
                "graph_id": result_graph_id,
                "graph_type": graph_type,
                "merge_strategy": merge_strategy if graph_type == "dynamic" else None,
                "filename": file.filename,
                "size_bytes": len(content),
            }

        finally:
            # Clean up temporary file
            try:
                os.unlink(temp_file.name)
            except (OSError, FileNotFoundError):
                # Temporary file cleanup failed, but this is not critical
                pass
            except Exception as e:
                logger.warning(f"Failed to cleanup temporary file: {str(e)}")

    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Error importing graph: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/graph/merge")
async def merge_graphs(
    request: Request,
    base_graph_id: str = Form(...),
    incoming_graph_id: str = Form(...),
    merge_strategy: str = Form("union"),
):
    """Merge one graph into another (dynamic graphs only)."""
    try:
        kg_manager = request.app.state.kg_manager
        if not kg_manager:
            raise HTTPException(
                status_code=HTTP_STATUS["SERVICE_UNAVAILABLE"],
                detail="Knowledge graph manager not initialized",
            )

        # Validate merge strategy
        if merge_strategy not in ["union", "prefer_base", "prefer_incoming"]:
            raise HTTPException(
                status_code=400,
                detail="merge_strategy must be 'union', 'prefer_base', or 'prefer_incoming'",
            )

        # This is a placeholder - would need to implement merge_indices method
        raise HTTPException(
            status_code=HTTP_STATUS["NOT_IMPLEMENTED"],
            detail="Graph merge functionality not implemented yet",
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error merging graphs: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/graph/{graph_id}/subgraph")
async def get_subgraph(
    request: Request,
    graph_id: str,
    query: str = Query(...),
    max_hops: int = Query(2),
    max_nodes: int = Query(100),
):
    """Extract a relevant subgraph for GraphRAG queries."""
    try:
        kg_manager = request.app.state.kg_manager
        if not kg_manager:
            raise HTTPException(
                status_code=HTTP_STATUS["SERVICE_UNAVAILABLE"],
                detail="Knowledge graph manager not initialized",
            )

        # Get subgraph (run in thread to avoid blocking)
        subgraph_data = await asyncio.to_thread(kg_manager.get_subgraph, query, max_hops, max_nodes)

        return {
            "graph_id": graph_id,
            "subgraph": subgraph_data,
            "timestamp": datetime.now().isoformat(),
        }

    except Exception as e:
        logger.error(f"Error getting subgraph: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/graph/import")
async def import_graph(request: Request, file: bytes = File(...), graph_id: Optional[str] = None):
    """Import a knowledge graph from a file."""
    try:
        kg_manager = request.app.state.kg_manager
        if not kg_manager:
            raise HTTPException(status_code=503, detail="Knowledge graph manager not initialized")

        raise HTTPException(
            status_code=501, detail="Graph import functionality not implemented yet"
        )
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error importing graph: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))
