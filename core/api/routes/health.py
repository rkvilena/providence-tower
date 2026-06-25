from __future__ import annotations

from fastapi import APIRouter, Depends

from core.api.dependencies import get_rag_service
from core.api.schemas import HealthResponse
from core.api.service import RagService

router = APIRouter(tags=["Health"])


@router.get("/health", response_model=HealthResponse)
async def health(
    svc: RagService = Depends(get_rag_service),
) -> HealthResponse:
    """Basic health-check endpoint."""
    redis_ok = False
    if svc.history_store is not None:
        try:
            svc.history_store.ping()
            redis_ok = True
        except Exception:
            redis_ok = False

    return HealthResponse(
        status="ok",
        redis_reachable=redis_ok,
        planner_llm_enabled=svc.is_planner_llm_enabled,
        reranker_enabled=svc.is_reranker_enabled,
        message="Providence Tower v2 RAG API is running.",
    )
