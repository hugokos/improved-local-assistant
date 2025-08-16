"""
WebSocket message models for the Improved Local AI Assistant.

This module defines Pydantic models for WebSocket message validation.
"""


from pydantic import BaseModel
from pydantic import validator
from services import DEFAULTS


class WSMessage(BaseModel):
    """WebSocket message model."""

    type: str
    content: str
    session_id: str | None = None

    @validator("content")
    def content_length(self, v):
        if len(v) > DEFAULTS["max_message_length"]:
            raise ValueError(f'Message too long (max {DEFAULTS["max_message_length"]} characters)')
        return v

    @validator("type")
    def valid_type(self, v):
        valid_types = ["message", "ping", "pong", "close"]
        if v not in valid_types:
            raise ValueError(f"Invalid message type. Must be one of: {valid_types}")
        return v
