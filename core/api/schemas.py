from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ConfigDict, Field


class ChatRequest(BaseModel):
    """Request payload for a RAG chat query."""

    model_config = ConfigDict(extra="forbid")

    query: str = Field(..., min_length=1, description="User query text")
    session_id: str | None = Field(
        default=None,
        description="Existing session ID to continue a conversation. Omit to auto-generate a new session.",
    )


class ChunkHitResponse(BaseModel):
    """A single chunk retrieved during the fetch phase."""

    model_config = ConfigDict(extra="forbid")

    chunk_id: str
    page_id: str
    page_title: str
    score: float
    text: str
    section: str | None = None
    subsection: str | None = None


class PlannerResponse(BaseModel):
    """Output of the planner node."""

    model_config = ConfigDict(extra="forbid")

    condensed_query: str = ""
    planned_queries: list[str] = Field(default_factory=list)
    entities: list[str] = Field(default_factory=list)
    reasoning: list[str] = Field(default_factory=list)
    source: str = "fallback_python"


class ChatResponse(BaseModel):
    """Response returned after a successful RAG invocation."""

    model_config = ConfigDict(extra="forbid")

    session_id: str
    response: str
    sufficient: bool = False
    used_chunk_ids: list[str] = Field(default_factory=list)
    debug_trace: list[str] = Field(default_factory=list)
    node_latencies_ms: dict[str, float] = Field(default_factory=dict)
    planner: PlannerResponse = Field(default_factory=PlannerResponse)
    chunk_count: int = 0


class ErrorResponse(BaseModel):
    """Standard error payload."""

    model_config = ConfigDict(extra="forbid")

    detail: str
    error_type: str = ""


class SessionClearResponse(BaseModel):
    """Response after clearing a session."""

    model_config = ConfigDict(extra="forbid")

    session_id: str
    cleared: bool = True


class HealthResponse(BaseModel):
    """Health-check payload."""

    model_config = ConfigDict(extra="forbid")

    status: str = "ok"
    redis_reachable: bool = False
    planner_llm_enabled: bool = False
    reranker_enabled: bool = False
    message: str = ""
