"""
WebSocket message models for the Improved Local AI Assistant.

This module defines Pydantic models for WebSocket message validation.
"""

from typing import Optional
from pydantic import BaseModel
from pydantic import field_validator
from improved_local_assistant.services import DEFAULTS


class WSMessage(BaseModel):
    """WebSocket message model."""

    type: str
    content: str
    session_id: Optional[str] = None

    @field_validator("content")
    @classmethod
    def content_length(cls, v):
        if len(v) > DEFAULTS["max_message_length"]:
            raise ValueError(f'Message too long (max {DEFAULTS["max_message_length"]} characters)')
        return v

    @field_validator("type")
    @classmethod
    def valid_type(cls, v):
        valid_types = ["message", "ping", "pong", "close"]
        if v not in valid_types:
            raise ValueError(f"Invalid message type. Must be one of: {valid_types}")
        return v
