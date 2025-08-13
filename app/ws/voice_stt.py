"""
Speech-to-Text WebSocket endpoint for the Improved Local AI Assistant.

This module provides WebSocket endpoints for real-time speech recognition
using the Vosk STT service.
"""

import json
import logging
import math
import struct
from datetime import datetime

from fastapi import WebSocket
from starlette.websockets import WebSocketDisconnect, WebSocketState

logger = logging.getLogger(__name__)


async def stt_websocket(websocket: WebSocket, session_id: str, app):
    """WebSocket endpoint for speech-to-text processing."""
    await websocket.accept()
    session_ready = False
    bytes_seen = 0
    
    # Get voice manager from app state
    voice_manager = getattr(app.state, "voice_manager", None)
    if not voice_manager:
        logger.error("Voice manager not available")
        await websocket.close(code=1011, reason="Voice services not initialized")
        return

    async def ensure_session():
        nonlocal session_ready
        if not session_ready:
            await voice_manager.create_voice_session(session_id)
            # recognizers get created lazily by VoiceManager/VoskSTTService on first use
            if websocket.client_state == WebSocketState.CONNECTED:
                await websocket.send_text(json.dumps({"type": "stt_ready"}))
            session_ready = True

    try:
        while True:
            msg = await websocket.receive()

            # Control messages (handshake)
            if "text" in msg and msg["text"] is not None:
                try:
                    data = json.loads(msg["text"])
                    if data.get("type") == "stt_start":
                        await ensure_session()
                except Exception:
                    pass
                continue

            # Binary audio frames
            if "bytes" in msg and msg["bytes"] is not None:
                await ensure_session()

                frame = msg["bytes"]
                bytes_seen += len(frame)

                # Optional: compute a quick RMS for UI
                # (same semantics as VoskSTTService._pcm16_stats)
                n = len(frame) // 2
                if n:
                    samples = struct.unpack("<" + "h" * n, frame[: n * 2])
                    rms = int(math.sqrt(sum(v * v for v in samples) / n))  # 0..32767
                    level = min(1.0, rms / 8000.0)  # simple normalization
                    if websocket.client_state == WebSocketState.CONNECTED:
                        await websocket.send_text(json.dumps({"type": "stt_level", "level": level}))

                # Feed audio to STT
                result = await voice_manager.process_audio_chunk(session_id, frame)

                # Relay recognition back to client
                if result.get("partial") and websocket.client_state == WebSocketState.CONNECTED:
                    await websocket.send_text(json.dumps({"type": "stt_partial", "text": result["partial"]}))
                elif result.get("final") and websocket.client_state == WebSocketState.CONNECTED:
                    await websocket.send_text(json.dumps({"type": "stt_final", "text": result["final"]}))

    except WebSocketDisconnect:
        pass  # client went away; do not try to send
    except Exception as e:
        logger.error(f"STT WebSocket error: {e}")
        try:
            if websocket.client_state == WebSocketState.CONNECTED:
                await websocket.close()
        except Exception:
            pass
    finally:
        # Cleanup
        if voice_manager and session_ready:
            voice_manager.set_voice_session_listening(session_id, False)
        logger.info(f"STT WebSocket closed for session {session_id}, processed {bytes_seen} bytes")

