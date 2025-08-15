"""
Text-to-Speech WebSocket endpoint for the Improved Local AI Assistant.

This module provides WebSocket endpoints for real-time speech synthesis
using the Piper TTS service.
"""

import asyncio
import json
import logging
import time
from datetime import datetime

from fastapi import WebSocket
from fastapi import WebSocketDisconnect
from starlette.websockets import WebSocketState

from app.core.websockets import ws_error

logger = logging.getLogger(__name__)


async def tts_websocket(websocket: WebSocket, session_id: str, app):
    """WebSocket endpoint for text-to-speech processing."""
    connection_start_time = time.time()
    client_info = {
        "ip": websocket.client.host,
        "session_id": session_id,
        "connection_time": datetime.now().isoformat(),
    }

    try:
        voice_manager = getattr(app.state, "voice_manager", None)

        if not voice_manager:
            logger.error("TTS WebSocket connect but voice manager not initialized")
            await websocket.accept()
            await ws_error(
                websocket,
                "Voice services not initialized. Please try again later.",
                code=1011,
                extra={"code": "VOICE_SERVICES_NOT_INITIALIZED"},
            )
            return

        await websocket.accept()
        logger.info(f"TTS WebSocket connected: {client_info}")

        # Create voice session if it doesn't exist
        if not await voice_manager.create_voice_session(session_id):
            await ws_error(
                websocket,
                "Failed to create voice session",
                code=1011,
                extra={"code": "VOICE_SESSION_CREATION_FAILED"},
            )
            return

        # Send ready message
        await websocket.send_json(
            {
                "type": "tts_ready",
                "session_id": session_id,
                "message": "Text-to-speech ready",
                "timestamp": datetime.now().isoformat(),
            }
        )

        try:
            while True:
                # Check connection state
                if websocket.application_state != WebSocketState.CONNECTED:
                    logger.info(f"TTS WebSocket no longer connected for session {session_id}")
                    break

                try:
                    # Receive text to synthesize or control messages
                    message_data = await websocket.receive_text()

                    try:
                        # Try to parse as JSON
                        message = json.loads(message_data)
                        text = message.get("text", "")
                        message_type = message.get("type", "synthesize")
                    except json.JSONDecodeError:
                        # Treat as plain text
                        text = message_data
                        message_type = "synthesize"

                    # Handle barge-in messages
                    if message_type == "barge_in":
                        logger.info(f"Barge-in request received for session {session_id}")
                        success = await voice_manager.handle_barge_in(session_id)

                        await websocket.send_json(
                            {
                                "type": "barge_in_ack",
                                "success": success,
                                "session_id": session_id,
                                "timestamp": datetime.now().isoformat(),
                            }
                        )
                        continue

                    # Handle synthesis requests
                    if not text.strip():
                        logger.warning(f"Empty text received for TTS in session {session_id}")
                        continue

                    logger.info(
                        f"TTS request for session {session_id}: '{text[:50]}{'...' if len(text) > 50 else ''}'"
                    )

                    # Ensure voice session exists
                    session_created = await voice_manager.create_voice_session(session_id)
                    logger.debug(
                        f"Voice session creation result for {session_id}: {session_created}"
                    )

                    if not session_created:
                        logger.error(f"Failed to create voice session {session_id}")
                        await websocket.send_json(
                            {
                                "type": "tts_error",
                                "message": "Failed to create voice session",
                                "session_id": session_id,
                                "timestamp": datetime.now().isoformat(),
                            }
                        )
                        continue

                    # Send synthesis start notification
                    await websocket.send_json(
                        {
                            "type": "tts_start",
                            "text": text,
                            "session_id": session_id,
                            "timestamp": datetime.now().isoformat(),
                        }
                    )

                    # Stream audio chunks
                    chunk_count = 0
                    total_bytes = 0
                    try:
                        logger.debug(f"Starting TTS synthesis for session {session_id}")
                        synthesis_generator = voice_manager.synthesize_response(session_id, text)
                        logger.debug(f"Got synthesis generator: {synthesis_generator}")

                        async for audio_chunk in synthesis_generator:
                            if websocket.application_state == WebSocketState.CONNECTED:
                                logger.debug(
                                    f"Sending audio chunk {chunk_count + 1}: {len(audio_chunk)} bytes, type: {type(audio_chunk)}"
                                )
                                await websocket.send_bytes(audio_chunk)
                                chunk_count += 1
                                total_bytes += len(audio_chunk)

                                # Debug log every 10 chunks
                                if chunk_count % 10 == 0:
                                    logger.debug(
                                        f"TTS sent {chunk_count} chunks, {total_bytes} bytes to session {session_id}"
                                    )
                                elif chunk_count == 1:
                                    logger.debug(
                                        f"TTS first chunk sent: {len(audio_chunk)} bytes to session {session_id}"
                                    )
                            else:
                                logger.info(
                                    f"TTS WebSocket disconnected during synthesis for session {session_id}"
                                )
                                break

                        # Send synthesis complete notification
                        if websocket.application_state == WebSocketState.CONNECTED:
                            await websocket.send_json(
                                {
                                    "type": "tts_end",
                                    "chunks_sent": chunk_count,
                                    "session_id": session_id,
                                    "timestamp": datetime.now().isoformat(),
                                }
                            )

                        logger.info(
                            f"TTS completed for session {session_id}: {chunk_count} chunks sent, {total_bytes} bytes total"
                        )

                    except Exception as e:
                        logger.error(
                            f"Error during TTS synthesis for session {session_id}: {str(e)}"
                        )
                        import traceback

                        logger.error(f"TTS synthesis traceback: {traceback.format_exc()}")

                        if websocket.application_state == WebSocketState.CONNECTED:
                            await websocket.send_json(
                                {
                                    "type": "tts_error",
                                    "message": "Speech synthesis failed",
                                    "error": str(e),
                                    "session_id": session_id,
                                    "timestamp": datetime.now().isoformat(),
                                }
                            )

                except WebSocketDisconnect:
                    logger.info(f"TTS WebSocket disconnected for session {session_id}")
                    break
                except Exception as e:
                    logger.error(f"Error processing TTS request: {str(e)}")
                    if websocket.application_state == WebSocketState.CONNECTED:
                        await ws_error(
                            websocket,
                            "TTS processing error",
                            extra={"details": str(e), "code": "TTS_PROCESSING_ERROR"},
                        )

        except WebSocketDisconnect:
            logger.info(f"TTS WebSocket disconnected for session {session_id}")
        except Exception as e:
            logger.error(f"Error in TTS WebSocket communication: {str(e)}")
            if websocket.application_state == WebSocketState.CONNECTED:
                await ws_error(
                    websocket,
                    "TTS WebSocket communication error",
                    extra={"details": str(e), "code": "TTS_WEBSOCKET_ERROR"},
                )

    except Exception as e:
        logger.error(f"Error handling TTS WebSocket connection: {str(e)}")
        try:
            if websocket.client_state != WebSocketState.DISCONNECTED:
                await websocket.close(code=1011, reason=f"TTS Internal error: {str(e)}")
        except Exception as close_error:
            logger.error(f"Error closing TTS WebSocket connection: {str(close_error)}")

    finally:
        logger.info(
            f"TTS WebSocket connection duration: {time.time() - connection_start_time:.2f}s"
        )


class TTSMessageQueue:
    """
    Queue for managing TTS messages to prevent overlap and ensure proper sequencing.
    """

    def __init__(self):
        self.queues = {}  # session_id -> asyncio.Queue
        self.active_sessions = set()

    async def add_message(self, session_id: str, text: str):
        """Add a message to the TTS queue for a session."""
        if session_id not in self.queues:
            self.queues[session_id] = asyncio.Queue()

        await self.queues[session_id].put(text)

    async def get_next_message(self, session_id: str) -> str:
        """Get the next message from the queue for a session."""
        if session_id not in self.queues:
            self.queues[session_id] = asyncio.Queue()

        return await self.queues[session_id].get()

    def mark_session_active(self, session_id: str):
        """Mark a session as actively processing TTS."""
        self.active_sessions.add(session_id)

    def mark_session_inactive(self, session_id: str):
        """Mark a session as no longer processing TTS."""
        self.active_sessions.discard(session_id)

    def is_session_active(self, session_id: str) -> bool:
        """Check if a session is actively processing TTS."""
        return session_id in self.active_sessions

    def cleanup_session(self, session_id: str):
        """Clean up resources for a session."""
        if session_id in self.queues:
            # Clear any remaining messages
            while not self.queues[session_id].empty():
                try:
                    self.queues[session_id].get_nowait()
                except asyncio.QueueEmpty:
                    break

            del self.queues[session_id]

        self.active_sessions.discard(session_id)


# Global TTS message queue
tts_queue = TTSMessageQueue()


async def queue_tts_message(session_id: str, text: str):
    """
    Queue a TTS message for processing.

    Args:
        session_id: Session identifier
        text: Text to synthesize
    """
    await tts_queue.add_message(session_id, text)


def is_tts_active(session_id: str) -> bool:
    """
    Check if TTS is currently active for a session.

    Args:
        session_id: Session identifier

    Returns:
        bool: True if TTS is active
    """
    return tts_queue.is_session_active(session_id)
