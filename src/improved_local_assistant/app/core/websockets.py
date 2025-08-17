"""
WebSocket utilities for the Improved Local AI Assistant.

This module provides helper functions and classes for WebSocket connections,
including a connection manager and safe send functions.
"""

import logging
from datetime import datetime
from typing import Any

from fastapi import WebSocket
from starlette.websockets import WebSocketDisconnect
from starlette.websockets import WebSocketState

logger = logging.getLogger(__name__)


async def ws_send_json_safe(ws: WebSocket, data: dict[str, Any]) -> bool:
    """
    Safely send JSON data over a WebSocket connection.

    Args:
        ws: WebSocket connection
        data: Data to send

    Returns:
        True if successful, False otherwise
    """
    try:
        if ws.application_state == WebSocketState.CONNECTED:
            await ws.send_json(data)
            return True
    except Exception as e:
        logger.error(f"Failed to send JSON over WebSocket: {str(e)}")
    return False


async def ws_send_text_safe(ws: WebSocket, text: str) -> bool:
    """
    Safely send text over a WebSocket connection.

    Args:
        ws: WebSocket connection
        text: Text to send

    Returns:
        True if successful, False otherwise
    """
    try:
        if ws.application_state == WebSocketState.CONNECTED:
            await ws.send_text(text)
            return True
    except Exception as e:
        logger.error(f"Failed to send text over WebSocket: {str(e)}")
    return False


async def ws_error(
    ws: WebSocket, message: str, code: int = 1011, extra: dict[str, Any] | None = None
):
    """
    Send an error over WebSocket and try to close gracefully.

    Args:
        ws: WebSocket connection
        message: Error message
        code: WebSocket close code
        extra: Additional data to include in the error message
    """
    payload = {"type": "error", "message": message}
    if extra:
        payload.update(extra)
    try:
        # Check connection state before sending
        if ws.application_state == WebSocketState.CONNECTED:
            await ws.send_json(payload)
    except (WebSocketDisconnect, RuntimeError):
        # Connection already closed, nothing to do
        pass
    except Exception:
        logger.exception("Failed to send WS error")
    finally:
        try:
            if ws.application_state == WebSocketState.CONNECTED:
                await ws.close(code=code, reason=message)
        except Exception:
            pass


class ConnectionManager:
    """Manage WebSocket connections."""

    def __init__(self):
        self.active_connections: dict[str, WebSocket] = {}
        self.connected_sockets: set = set()  # Track live sockets

    async def connect(self, websocket: WebSocket, session_id: str):
        """
        Accept a WebSocket connection and store it.

        Args:
            websocket: WebSocket connection
            session_id: Session identifier
        """
        await websocket.accept()
        self.active_connections[session_id] = websocket
        self.connected_sockets.add(websocket)
        logger.info(f"WebSocket connected for session {session_id}")

    def disconnect(self, session_id: str):
        """
        Remove a WebSocket connection.

        Args:
            session_id: Session identifier
        """
        if session_id in self.active_connections:
            websocket = self.active_connections[session_id]
            self.connected_sockets.discard(websocket)
            del self.active_connections[session_id]
            logger.info(f"WebSocket disconnected for session {session_id}")

    async def send_text(self, session_id: str, message: str):
        """
        Send text to a specific WebSocket connection.

        Args:
            session_id: Session identifier
            message: Text message to send
        """
        ws = self.active_connections.get(session_id)
        if ws:
            await ws.send_text(message)

    async def send_json(self, session_id: str, data: dict[str, Any]):
        """
        Send JSON data to a specific WebSocket connection.

        Args:
            session_id: Session identifier
            data: JSON data to send
        """
        ws = self.active_connections.get(session_id)
        if ws and ws in self.connected_sockets:
            try:
                if ws.application_state == WebSocketState.CONNECTED:
                    await ws.send_json(data)
                else:
                    self.disconnect(session_id)
            except (WebSocketDisconnect, RuntimeError):
                self.disconnect(session_id)
            except Exception:
                logger.exception(f"Failed to send JSON to session {session_id}")
                self.disconnect(session_id)

    async def broadcast_json(self, data: dict[str, Any]):
        """
        Broadcast JSON data to all active WebSocket connections.

        Args:
            data: JSON data to broadcast
        """
        dead = set()
        for ws in self.connected_sockets:
            if ws.application_state != WebSocketState.CONNECTED:
                dead.add(ws)
                continue
            try:
                await ws.send_json(data)
            except (WebSocketDisconnect, RuntimeError):
                dead.add(ws)
            except Exception as e:
                logger.error(f"Error broadcasting to WebSocket: {str(e)}")
                dead.add(ws)

        # Clean up disconnected sockets
        self.connected_sockets.difference_update(dead)

        # Clean up active_connections mapping
        disconnected_sessions = []
        for sid, ws in list(self.active_connections.items()):
            if ws in dead:
                disconnected_sessions.append(sid)

        for sid in disconnected_sessions:
            self.disconnect(sid)

    async def notify_graph_update(self, graph_id: str, update_type: str):
        """
        Notify all clients about a graph update.

        Args:
            graph_id: Graph identifier
            update_type: Type of update (created, updated, deleted)
        """
        await self.broadcast_json(
            {
                "type": "graph_update",
                "graph_id": graph_id,
                "update_type": update_type,
                "timestamp": datetime.now().isoformat(),
            }
        )
