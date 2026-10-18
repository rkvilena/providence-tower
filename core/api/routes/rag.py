from __future__ import annotations

import logging

from fastapi import APIRouter, Depends
from fastapi.responses import StreamingResponse

from core.api.rate_limiter import get_redis_rate_limiter

from core.api.dependencies import get_rag_service
from core.api.schemas import (
    ChatRequest,
    ChatResponse,
    ErrorResponse,
    SessionClearResponse,
    build_chat_response,
)
from core.api.service import RagService

LOGGER = logging.getLogger(__name__)

router = APIRouter(prefix="/rag", tags=["RAG"])


@router.post(
    "/chat",
    response_model=ChatResponse,
    responses={
        422: {"model": ErrorResponse},
        429: {"model": ErrorResponse},
    },
    summary="Send a query through the full RAG pipeline",
)
def chat(
    body: ChatRequest,
    svc: RagService = Depends(get_rag_service),
    _rate_limiter: None = Depends(get_redis_rate_limiter),
) -> ChatResponse:
    """Execute the planner → fetcher → thinker → context pipeline.

    If ``session_id`` is provided the conversation history is loaded from Redis
    (if available) and the Q&A turn is persisted afterwards.  When omitted a
    brand-new session is created.
    """
    result = svc.run_rag(query=body.query, session_id=body.session_id)
    return build_chat_response(result)


@router.post(
    "/chat/stream",
    summary="Stream RAG phase progress and the final answer",
    responses={429: {"model": ErrorResponse}},
)
def chat_stream(
    body: ChatRequest,
    svc: RagService = Depends(get_rag_service),
    _rate_limiter: None = Depends(get_redis_rate_limiter),
) -> StreamingResponse:
    """Stream lifecycle events while the synchronous RAG pipeline executes.

    Emits ``text/event-stream`` frames (``event: <name>`` + ``data: <json>``)
    as the graph moves through its nodes, then a ``final_answer`` event with
    the same payload as ``/rag/chat``.  Unrecoverable failures are reported
    as a ``pipeline_error`` event instead of an HTTP error status.
    """

    def events():
        for event in svc.stream_rag(query=body.query, session_id=body.session_id):
            yield event.encode()

    return StreamingResponse(
        events(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )


@router.post(
    "/session/clear",
    response_model=SessionClearResponse,
    responses={422: {"model": ErrorResponse}},
    summary="Clear conversation history for a session",
)
def clear_session(
    body: ChatRequest,
    svc: RagService = Depends(get_rag_service),
) -> SessionClearResponse:
    """Delete the stored history for a given session.

    The ``body.query`` field is ignored; only ``session_id`` is used.  If
    ``session_id`` is *None* a new session ID is returned without clearing
    anything.
    """
    if not body.session_id or not str(body.session_id).strip():
        from core.rag.history import generate_session_id

        return SessionClearResponse(
            session_id=generate_session_id(),
            cleared=True,
        )

    sid = str(body.session_id).strip()
    svc.clear_session(sid)
    return SessionClearResponse(session_id=sid, cleared=True)
