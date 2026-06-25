from __future__ import annotations

from fastapi import Request

from core.api.service import RagService


def get_rag_service(request: Request) -> RagService:
    """FastAPI dependency: retrieve the application-scoped ``RagService``.

    Usage::

        @router.post("/chat")
        async def chat(
            body: ChatRequest,
            svc: RagService = Depends(get_rag_service),
        ) -> ChatResponse: ...
    """
    svc: RagService | None = getattr(request.app.state, "rag_service", None)
    if svc is None:
        raise RuntimeError("RagService not initialised – application not started?")
    return svc
