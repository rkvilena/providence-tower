from __future__ import annotations

import logging
from typing import Any

from fastapi import APIRouter, Depends

from core.api.rate_limiter import get_redis_rate_limiter

from core.api.dependencies import get_rag_service
from core.api.schemas import (
    ChatRequest,
    ChatResponse,
    ErrorResponse,
    PlannerResponse,
    SessionClearResponse,
)
from core.api.service import RagService
from core.rag.schema import RagState

LOGGER = logging.getLogger(__name__)

router = APIRouter(prefix="/rag", tags=["RAG"])


def _build_chat_response(state: RagState) -> ChatResponse:
    """Convert a final ``RagState`` into the public ``ChatResponse`` schema."""
    think = state.thinker_state
    plan = state.planner_state

    return ChatResponse(
        session_id=state.session_id,
        response=(think.response or "").strip(),
        sufficient=think.sufficient,
        used_chunk_ids=list(think.used_chunk_ids),
        debug_trace=list(state.debug_trace),
        node_latencies_ms=dict(state.node_latencies_ms),
        planner=PlannerResponse(
            condensed_query=plan.condensed_query,
            planned_queries=list(plan.planned_queries),
            entities=list(plan.entities),
            reasoning=list(plan.reasoning),
            source=plan.source,
        ),
        chunk_count=len(state.fetcher_state.chunks),
    )


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
    return _build_chat_response(result)


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
