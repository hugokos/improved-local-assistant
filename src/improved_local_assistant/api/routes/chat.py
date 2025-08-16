"""
Chat API endpoints for the Improved Local AI Assistant.

This module provides REST API endpoints for chat functionality.
"""

import logging

from app.schemas import MessageRequest
from fastapi import APIRouter
from fastapi import HTTPException
from fastapi import Request

router = APIRouter()
logger = logging.getLogger(__name__)


@router.post("/chat")
async def chat(request: Request, message_request: MessageRequest):
    """REST API endpoint for chat messages."""
    try:
        conversation_manager = request.app.state.conversation_manager
        if not conversation_manager:
            raise HTTPException(status_code=503, detail="Conversation manager not initialized")

        session_id = message_request.session_id
        if not session_id or session_id not in conversation_manager.sessions:
            session_id = conversation_manager.create_session()

        response = ""
        async for token in conversation_manager.converse_with_context(
            session_id, message_request.message
        ):
            response += token

        return {"response": response, "session_id": session_id}
    except Exception as e:
        logger.error(f"Error in chat endpoint: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/sessions")
async def list_sessions(request: Request):
    """List all active conversation sessions."""
    try:
        conversation_manager = request.app.state.conversation_manager
        if not conversation_manager:
            raise HTTPException(status_code=503, detail="Conversation manager not initialized")

        sessions = conversation_manager.list_sessions()
        return {"sessions": sessions, "count": len(sessions)}
    except Exception as e:
        logger.error(f"Error listing sessions: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/session/{session_id}")
async def get_session(request: Request, session_id: str):
    """Get information about a specific session."""
    try:
        conversation_manager = request.app.state.conversation_manager
        if not conversation_manager:
            raise HTTPException(status_code=503, detail="Conversation manager not initialized")

        session_info = conversation_manager.get_session_info(session_id)
        if "error" in session_info:
            raise HTTPException(status_code=404, detail=session_info["error"])

        return session_info
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting session info: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@router.delete("/session/{session_id}")
async def delete_session(request: Request, session_id: str):
    """Delete a conversation session."""
    try:
        conversation_manager = request.app.state.conversation_manager
        if not conversation_manager:
            raise HTTPException(status_code=503, detail="Conversation manager not initialized")

        success = conversation_manager.delete_session(session_id)
        if not success:
            raise HTTPException(status_code=404, detail=f"Session {session_id} not found")

        return {"status": "success", "message": f"Session {session_id} deleted"}
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error deleting session: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/session/{session_id}/citations")
async def get_session_citations(request: Request, session_id: str):
    """Get citations for the last query in a session."""
    try:
        conversation_manager = request.app.state.conversation_manager
        if not conversation_manager:
            raise HTTPException(status_code=503, detail="Conversation manager not initialized")

        citations_data = conversation_manager.get_citations(session_id)
        if "error" in citations_data:
            raise HTTPException(status_code=404, detail=citations_data["error"])

        return citations_data
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting citations: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))
