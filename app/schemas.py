"""
Pydantic schemas for the Improved Local AI Assistant API.

This module defines the request and response models for the FastAPI endpoints.
"""

from typing import Optional, List, Dict, Any
from pydantic import BaseModel, Field


class MessageRequest(BaseModel):
    """Request model for chat messages."""
    message: str = Field(..., description="The user's message")
    session_id: Optional[str] = Field(None, description="Session ID for conversation continuity")


class MessageResponse(BaseModel):
    """Response model for chat messages."""
    response: str = Field(..., description="The assistant's response")
    session_id: str = Field(..., description="Session ID for conversation continuity")


class GraphRequest(BaseModel):
    """Request model for knowledge graph queries."""
    query: str = Field(..., description="The query to search in knowledge graphs")
    graph_ids: Optional[List[str]] = Field(None, description="Specific graph IDs to search (optional)")


class GraphResponse(BaseModel):
    """Response model for knowledge graph queries."""
    response: str = Field(..., description="The response from knowledge graphs")
    source_nodes: List[str] = Field(default_factory=list, description="Source nodes used in the response")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional metadata")


class GraphCreateRequest(BaseModel):
    """Request model for creating new knowledge graphs."""
    docs_path: str = Field(..., description="Path to documents for graph creation")
    graph_id: Optional[str] = Field(None, description="Optional ID for the new graph")


class GraphTraversalRequest(BaseModel):
    """Request model for graph traversal operations."""
    source: str = Field(..., description="Source node for traversal")
    target: str = Field(..., description="Target node for traversal")
    max_hops: int = Field(default=3, description="Maximum number of hops for traversal")


class SessionInfo(BaseModel):
    """Model for session information."""
    session_id: str = Field(..., description="Session ID")
    created_at: str = Field(..., description="Session creation timestamp")
    updated_at: str = Field(..., description="Last update timestamp")
    message_count: int = Field(..., description="Number of messages in session")
    has_summary: bool = Field(..., description="Whether session has a summary")


class HealthResponse(BaseModel):
    """Response model for health check."""
    status: str = Field(..., description="Overall system status")
    timestamp: str = Field(..., description="Health check timestamp")
    services: Dict[str, str] = Field(..., description="Status of individual services")
    models: Dict[str, Any] = Field(..., description="Model information")
    knowledge_graphs: Dict[str, int] = Field(..., description="Knowledge graph statistics")
    system: Dict[str, Any] = Field(..., description="System information")
    version: str = Field(..., description="Application version")


class MetricsResponse(BaseModel):
    """Response model for system metrics."""
    timestamp: str = Field(..., description="Metrics timestamp")
    resource_usage: Dict[str, Any] = Field(..., description="Resource usage metrics")
    performance_metrics: Dict[str, Any] = Field(..., description="Performance metrics")
    model_metrics: Dict[str, Any] = Field(..., description="Model-specific metrics")
    kg_metrics: Dict[str, Any] = Field(..., description="Knowledge graph metrics")
    conversation_metrics: Dict[str, Any] = Field(..., description="Conversation metrics")


class WSMessage(BaseModel):
    """WebSocket message model."""
    type: str = Field(..., description="Message type")
    content: str = Field(..., description="Message content")
    session_id: str = Field(..., description="Session ID")
    message_id: Optional[str] = Field(None, description="Message ID")
    timestamp: Optional[str] = Field(None, description="Message timestamp")