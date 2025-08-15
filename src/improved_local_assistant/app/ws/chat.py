"""
Chat WebSocket endpoint for the Improved Local AI Assistant.

This module provides WebSocket endpoints for real-time chat functionality.
"""

import asyncio
import logging
import time
import uuid
from datetime import datetime

from fastapi import WebSocket
from fastapi import WebSocketDisconnect
from services import get_timeout
from starlette.websockets import WebSocketState

from app.core.websockets import ws_error

logger = logging.getLogger(__name__)


async def chat_websocket(websocket: WebSocket, session_id: str, app):
    """WebSocket endpoint for real-time chat."""
    connection_start_time = time.time()
    client_info = {
        "ip": websocket.client.host,
        "session_id": session_id,
        "connection_time": datetime.now().isoformat(),
    }

    try:
        conversation_manager = getattr(app.state, "conversation_manager", None)
        kg_manager = getattr(app.state, "kg_manager", None)
        system_monitor = getattr(app.state, "system_monitor", None)
        connection_manager = getattr(app.state, "connection_manager", None)
        dynamic_model_manager = getattr(app.state, "dynamic_model_manager", None)
        config = getattr(app.state, "config", {})

        if not conversation_manager:
            logger.error("WS connect but services not initialized")
            await websocket.accept()
            await ws_error(
                websocket,
                "Services not initialized. Please try again later.",
                code=1011,
                extra={"code": "SERVICES_NOT_INITIALIZED"},
            )
            return

        await connection_manager.connect(websocket, session_id)
        logger.info(f"WebSocket connected: {client_info}")

        # Register WebSocket with dynamic model manager for notifications
        if dynamic_model_manager:
            dynamic_model_manager.register_websocket(websocket)

        # Ensure session exists in conversation manager
        if session_id not in conversation_manager.sessions:
            # Create session with the same ID as the WebSocket session
            conversation_manager.sessions[session_id] = {
                "messages": [],
                "created_at": datetime.now(),
                "updated_at": datetime.now(),
                "summary": None,
                "metadata": {"total_messages": 0, "user_messages": 0, "assistant_messages": 0},
            }
            conversation_manager.metrics["sessions_created"] += 1
            logger.info(f"Created conversation session: {session_id}")

            await websocket.send_json(
                {
                    "type": "system",
                    "message": "Connected to Improved Local AI Assistant",
                    "session_id": session_id,
                    "status": "ready",
                }
            )
        else:
            session_info = conversation_manager.get_session_info(session_id)
            await websocket.send_json(
                {
                    "type": "system",
                    "message": "Reconnected to existing session",
                    "session_id": session_id,
                    "session_info": session_info,
                    "status": "ready",
                }
            )

        if system_monitor:
            await websocket.send_json(
                {
                    "type": "system_status",
                    "resource_usage": system_monitor.get_resource_usage(),
                    "health": system_monitor.check_health(),
                    "timestamp": datetime.now().isoformat(),
                }
            )

        heartbeat_interval = get_timeout("ws_heartbeat", config)
        last_heartbeat = time.time()

        try:
            while True:
                # Check connection state before any operation
                if websocket.application_state != WebSocketState.CONNECTED:
                    logger.info(f"WebSocket no longer connected for session {session_id}")
                    break

                # heartbeat
                now = time.time()
                if now - last_heartbeat >= heartbeat_interval:
                    try:
                        if websocket.application_state == WebSocketState.CONNECTED:
                            await websocket.send_json(
                                {"type": "heartbeat", "timestamp": datetime.now().isoformat()}
                            )
                            last_heartbeat = now
                    except (WebSocketDisconnect, RuntimeError):
                        logger.info(
                            f"WebSocket disconnected during heartbeat for session {session_id}"
                        )
                        break

                # receive with timeout
                try:
                    message = await asyncio.wait_for(
                        websocket.receive_text(), timeout=get_timeout("ws_receive_timeout", config)
                    )
                except asyncio.TimeoutError:
                    continue
                except WebSocketDisconnect:
                    logger.info(f"WebSocket disconnected during receive for session {session_id}")
                    break
                except RuntimeError as e:
                    if "disconnect message has been received" in str(e):
                        logger.info(f"WebSocket already disconnected for session {session_id}")
                        break
                    else:
                        logger.error(f"Runtime error in WebSocket receive: {str(e)}")
                        break
                except Exception as e:
                    logger.error(f"Error receiving WS message: {str(e)}")
                    if websocket.application_state == WebSocketState.CONNECTED:
                        await ws_error(
                            websocket,
                            "Communication error",
                            extra={"details": str(e), "code": "WEBSOCKET_RECEIVE_ERROR"},
                        )
                    break

                # Handle plain text message from client
                try:
                    user_message = message
                    message_id = str(uuid.uuid4())
                    logger.info(
                        f"Received message from session {session_id}: {message_id} (len: {len(user_message)})"
                    )

                    # Validate message is not empty
                    if not user_message.strip():
                        logger.warning(f"Empty message from session {session_id}")
                        await ws_error(
                            websocket,
                            "Empty message received",
                            code=1003,
                            extra={"code": "EMPTY_MESSAGE"},
                        )
                        continue

                except Exception as e:
                    logger.warning(
                        f"Error processing WebSocket message from session {session_id}: {str(e)}"
                    )
                    await ws_error(
                        websocket,
                        f"Invalid message: {str(e)}",
                        code=1003,
                        extra={"code": "MESSAGE_ERROR"},
                    )
                    continue

                # Send typing indicator only if connected
                if websocket.application_state == WebSocketState.CONNECTED:
                    try:
                        await websocket.send_json(
                            {"type": "typing", "status": "start", "message_id": message_id}
                        )
                    except (WebSocketDisconnect, RuntimeError):
                        logger.info(
                            f"WebSocket disconnected during typing start for session {session_id}"
                        )
                        break

                try:
                    if system_monitor and system_monitor.metrics["system"]["cpu_percent"] > 80:
                        await websocket.send_json(
                            {
                                "type": "warning",
                                "message": "System is under high load. Response may be delayed.",
                                "message_id": message_id,
                            }
                        )

                    process_start_time = time.time()
                    response_text = ""

                    async def process_with_timeout():
                        nonlocal response_text
                        async for token in conversation_manager.converse_with_context(
                            session_id, user_message
                        ):
                            response_text += token
                            # Check connection before sending each token
                            if websocket.application_state == WebSocketState.CONNECTED:
                                await websocket.send_text(token)
                            else:
                                break

                    await asyncio.wait_for(
                        process_with_timeout(), timeout=get_timeout("conversation_timeout", config)
                    )

                    logger.info(
                        f"Processed message {message_id} in {time.time() - process_start_time:.2f}s"
                    )

                except asyncio.TimeoutError:
                    logger.error(f"Message processing timed out for session {session_id}")
                    if websocket.application_state == WebSocketState.CONNECTED:
                        await websocket.send_json(
                            {
                                "type": "error",
                                "message": "Processing timed out. The response was taking too long.",
                                "code": "PROCESSING_TIMEOUT",
                                "message_id": message_id,
                            }
                        )
                except Exception as e:
                    logger.error(f"Error processing message: {str(e)}")
                    error_info = {
                        "type": "error",
                        "message": "Error processing your message",
                        "details": str(e),
                        "code": "PROCESSING_ERROR",
                        "message_id": message_id,
                        "timestamp": datetime.now().isoformat(),
                    }
                    if "session not found" in str(e).lower():
                        error_info[
                            "recovery"
                        ] = "Your session may have expired. Try refreshing the page."
                    elif "model" in str(e).lower() and "not available" in str(e).lower():
                        error_info[
                            "recovery"
                        ] = "The AI model is currently unavailable. Please try again later."

                    if websocket.application_state == WebSocketState.CONNECTED:
                        await websocket.send_json(error_info)

                # Send typing stop indicator
                if websocket.application_state == WebSocketState.CONNECTED:
                    try:
                        await websocket.send_json(
                            {"type": "typing", "status": "stop", "message_id": message_id}
                        )
                    except (WebSocketDisconnect, RuntimeError):
                        logger.info(
                            f"WebSocket disconnected during typing stop for session {session_id}"
                        )
                        break

                # Send citations and dynamic KG updates
                if conversation_manager and websocket.application_state == WebSocketState.CONNECTED:
                    try:
                        # Send citations
                        citations_data = conversation_manager.get_citations(session_id)
                        logger.info(
                            f"📚 Retrieved citations data: {len(citations_data.get('citations', []))} citations"
                        )

                        if citations_data.get("citations"):
                            await websocket.send_json(
                                {
                                    "type": "citations",
                                    "data": citations_data,
                                    "message_id": message_id,
                                }
                            )
                            logger.info(
                                f"✅ Sent {len(citations_data['citations'])} citations to WebSocket"
                            )
                        else:
                            await websocket.send_json(
                                {
                                    "type": "citations",
                                    "data": {
                                        "citations": [],
                                        "message": "No citations available for this response",
                                    },
                                    "message_id": message_id,
                                }
                            )
                            logger.info("📭 Sent empty citations message to WebSocket")

                        # Send dynamic KG updates
                        dynamic_triples = conversation_manager.get_dynamic_triples(session_id)
                        logger.info(f"🔗 Retrieved dynamic triples: {len(dynamic_triples)} triples")

                        if dynamic_triples:
                            await websocket.send_json(
                                {
                                    "type": "dynamic_kg_update",
                                    "data": {"triples": dynamic_triples},
                                    "message_id": message_id,
                                }
                            )
                            logger.info(
                                f"✅ Sent {len(dynamic_triples)} dynamic triples to WebSocket"
                            )
                            for triple in dynamic_triples:
                                logger.info(
                                    f"   📊 {triple.get('subject', 'N/A')} -> {triple.get('predicate', 'N/A')} -> {triple.get('object', 'N/A')}"
                                )
                        else:
                            logger.info("📭 No dynamic triples to send")

                    except (WebSocketDisconnect, RuntimeError):
                        logger.info(
                            f"WebSocket disconnected during citations send for session {session_id}"
                        )
                        break
                    except Exception as e:
                        logger.error(f"Error getting citations: {str(e)}")
                        try:
                            await websocket.send_json(
                                {
                                    "type": "warning",
                                    "message": "Could not retrieve citations",
                                    "details": str(e),
                                    "message_id": message_id,
                                }
                            )
                        except (WebSocketDisconnect, RuntimeError):
                            break

                # post-process system status
                if system_monitor:
                    try:
                        # Check if WebSocket is still connected before sending
                        if websocket.application_state == WebSocketState.CONNECTED:
                            await websocket.send_json(
                                {
                                    "type": "system_status",
                                    "resource_usage": system_monitor.get_resource_usage(),
                                    "timestamp": datetime.now().isoformat(),
                                }
                            )
                    except Exception as e:
                        logger.error(f"Error sending system status: {str(e)}")

        except WebSocketDisconnect:
            logger.info(f"WebSocket disconnected for session {session_id}")
            if connection_manager:
                connection_manager.disconnect(session_id)
            # Unregister from dynamic model manager
            if dynamic_model_manager:
                dynamic_model_manager.unregister_websocket(websocket)
            logger.info(
                f"WebSocket connection duration: {time.time() - connection_start_time:.2f}s"
            )

        except Exception as e:
            logger.error(f"Error in WebSocket communication: {str(e)}")
            if websocket.application_state == WebSocketState.CONNECTED:
                await ws_error(
                    websocket,
                    "WebSocket communication error",
                    extra={"details": str(e), "code": "WEBSOCKET_ERROR"},
                )
            if connection_manager:
                connection_manager.disconnect(session_id)
            # Unregister from dynamic model manager
            if dynamic_model_manager:
                dynamic_model_manager.unregister_websocket(websocket)

    except Exception as e:
        logger.error(f"Error handling WebSocket connection: {str(e)}")
        try:
            if websocket.client_state != WebSocketState.DISCONNECTED:
                await websocket.close(code=1011, reason=f"Internal error: {str(e)}")
        except Exception as close_error:
            logger.error(f"Error closing WebSocket connection: {str(close_error)}")
            try:
                await websocket.send_json({"type": "error", "message": str(e)})
            except Exception:
                pass
