"""
Monitor WebSocket endpoint for the Improved Local AI Assistant.

This module provides WebSocket endpoints for real-time system monitoring.
"""

import asyncio
import contextlib
import logging
from datetime import datetime

from fastapi import WebSocket
from fastapi import WebSocketDisconnect
from improved_local_assistant.services import DEFAULTS

logger = logging.getLogger(__name__)


async def monitor_websocket(websocket: WebSocket, app):
    """WebSocket endpoint for real-time system monitoring."""
    try:
        system_monitor = getattr(app.state, "system_monitor", None)
        if not system_monitor:
            await websocket.close(code=1011, reason="System monitor not initialized")
            return

        await websocket.accept()

        await websocket.send_json(
            {
                "type": "system_status",
                "resource_usage": system_monitor.get_resource_usage(),
                "health": system_monitor.check_health(),
                "timestamp": datetime.now().isoformat(),
            }
        )

        try:
            while True:
                await asyncio.sleep(DEFAULTS["monitoring_interval"])
                await websocket.send_json(
                    {
                        "type": "system_status",
                        "resource_usage": system_monitor.get_resource_usage(),
                        "health": system_monitor.check_health(),
                        "timestamp": datetime.now().isoformat(),
                    }
                )
        except WebSocketDisconnect:
            logger.info("Monitor WebSocket disconnected")
        except Exception as e:
            logger.error(f"Error in monitor WebSocket loop: {str(e)}")
    except Exception as e:
        logger.error(f"Error in monitor WebSocket: {str(e)}")
        with contextlib.suppress(Exception):
            await websocket.close(code=1011, reason=str(e))
