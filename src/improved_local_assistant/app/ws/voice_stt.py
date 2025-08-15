"""
Speech-to-Text WebSocket endpoint for the Improved Local AI Assistant.

This module provides WebSocket endpoints for real-time speech recognition
using the Vosk STT service.
"""

import array
import logging
import math

from fastapi import WebSocket
from fastapi import WebSocketDisconnect

logger = logging.getLogger(__name__)


def rms16le(b: bytes) -> float:
    """Calculate RMS of 16-bit little-endian PCM audio."""
    if not b:
        return 0.0
    a = array.array("h")  # 16-bit signed
    a.frombytes(b)
    return math.sqrt(sum(x * x for x in a) / len(a))


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

    try:
        # 1) Handshake (text frame)
        msg = await websocket.receive_json()  # e.g., {type:"stt_start", ...}
        if msg.get("type") == "stt_start":
            await voice_manager.create_voice_session(session_id)
            await websocket.send_json({"type": "stt_ready"})
            session_ready = True
            logger.info(f"STT session {session_id} ready")

        # 2) Audio loop (binary frames) - use iter_bytes for robust binary handling
        logger.info(f"🎵 Starting audio loop for session {session_id}")
        frame_count = 0

        async for chunk in websocket.iter_bytes():
            if not chunk:
                continue

            frame_count += 1
            bytes_seen += len(chunk)

            # Calculate RMS for speech detection
            r = rms16le(chunk)

            # Only log speech frames and occasional silence frames
            if r >= 50:
                logger.info(f"🎤 Speech frame {frame_count}: len={len(chunk)}, RMS={r:.1f}")
            elif frame_count % 50 == 0:  # Log every 50th silence frame
                logger.debug(f"🔇 Silence frame {frame_count}: len={len(chunk)}, RMS={r:.1f}")

            # Feed to STT service
            result = await voice_manager.process_audio_chunk(session_id, chunk)

            # Send results back to client
            if result.get("partial"):
                logger.info(f"📝 Partial result: '{result['partial']}'")
                await websocket.send_json({"type": "stt_partial", "text": result["partial"]})
            if result.get("final"):
                logger.info(f"✅ Final result: '{result['final']}'")
                await websocket.send_json({"type": "stt_final", "text": result["final"]})

    except WebSocketDisconnect:
        logger.info(f"STT WebSocket disconnected for session {session_id}")
    except Exception as e:
        logger.error(f"STT WebSocket error: {e}")
    finally:
        # Cleanup
        if voice_manager and session_ready:
            voice_manager.set_voice_session_listening(session_id, False)
        logger.info(f"STT WebSocket closed for session {session_id}, processed {bytes_seen} bytes")
