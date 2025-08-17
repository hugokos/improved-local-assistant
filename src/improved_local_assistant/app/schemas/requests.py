"""
Request models for the Improved Local AI Assistant API.

This module defines Pydantic models for API request validation.
"""

from pydantic import BaseModel


class MessageRequest(BaseModel):
    """Request model for chat messages."""

    message: str
    session_id: str | None = None


class GraphRequest(BaseModel):
    """Request model for graph queries."""

    query: str
    session_id: str | None = None


class GraphCreateRequest(BaseModel):
    """Request model for graph creation."""

    docs_path: str
    graph_id: str | None = None


class GraphTraversalRequest(BaseModel):
    """Request model for graph traversal."""

    source: str
    target: str
    max_hops: int = 3
