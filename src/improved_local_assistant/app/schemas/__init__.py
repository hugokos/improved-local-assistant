"""Pydantic models for the Improved Local AI Assistant."""

from improved_local_assistant.app.schemas.requests import GraphCreateRequest
from improved_local_assistant.app.schemas.requests import GraphRequest
from improved_local_assistant.app.schemas.requests import GraphTraversalRequest
from improved_local_assistant.app.schemas.requests import MessageRequest
from improved_local_assistant.app.schemas.websocket import WSMessage

__all__ = [
    "MessageRequest",
    "GraphRequest",
    "GraphCreateRequest",
    "GraphTraversalRequest",
    "WSMessage",
]
