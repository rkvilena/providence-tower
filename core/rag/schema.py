from __future__ import annotations

from datetime import datetime, timezone
from typing import Literal, TypedDict

from pydantic import BaseModel, ConfigDict, Field


def now_utc_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


class Message(BaseModel):
    model_config = ConfigDict(extra="forbid")

    role: str
    content: str
    timestamp: str = Field(default_factory=now_utc_iso)


class GraphState(TypedDict):
    state: RagState


class HistoryTurn(BaseModel):
    model_config = ConfigDict(extra="forbid")

    q: str
    a: str


# Planner schema
class PlannerState(BaseModel):
    model_config = ConfigDict(extra="forbid")

    condensed_query: str = ""
    planned_queries: list[str] = Field(default_factory=list)
    entities: list[str] = Field(default_factory=list)
    reasoning: list[str] = Field(default_factory=list)
    source: Literal["llm", "manual"] = "manual"


class ChunkHit(BaseModel):
    model_config = ConfigDict(extra="forbid")

    chunk_id: str
    page_id: str
    page_title: str
    score: float
    text: str
    section: str | None = None
    subsection: str | None = None


# Fetcher schema
class QueryFetchTiming(BaseModel):
    model_config = ConfigDict(extra="forbid")

    query: str = ""
    embedding_ms: float = 0.0
    vector_search_ms: float = 0.0
    hit_conversion_ms: float = 0.0
    rerank_ms: float = 0.0
    merge_ms: float = 0.0
    result_count: int = 0
    rerank_passed: bool | None = None


class FetchTimingTotals(BaseModel):
    model_config = ConfigDict(extra="forbid")

    embedding_ms: float = 0.0
    vector_search_ms: float = 0.0
    hit_conversion_ms: float = 0.0
    rerank_ms: float = 0.0
    merge_ms: float = 0.0
    final_sort_ms: float = 0.0
    query_count: int = 0
    unique_chunks: int = 0
    rerank_pair_count: int = 0
    rerank_model_calls: int = 0


class FetcherState(BaseModel):
    model_config = ConfigDict(extra="forbid")

    chunks: list[ChunkHit] = Field(default_factory=list)
    query_timings: list[QueryFetchTiming] = Field(default_factory=list)
    totals: FetchTimingTotals | None = None


# Thinker schema
class ThinkerState(BaseModel):
    model_config = ConfigDict(extra="forbid")

    sufficient: bool = False
    used_chunk_ids: list[str] = Field(default_factory=list)
    response: str = ""
    reasoning: list[str] = Field(default_factory=list)
    """Deprecated: no consumer reads this. Kept optional for backward compat;
    the prompt instructs the model to return an empty list."""


class ThinkerTiming(BaseModel):
    model_config = ConfigDict(extra="forbid")

    latency_ms: float = 0.0
    input_chunk_count: int = 0
    input_chars: int = 0
    estimated_input_tokens: int = 0
    output_chars: int = 0
    estimated_output_tokens: int = 0
    used_chunk_count: int = 0
    dropped_chunk_count: int = 0
    truncated_chunk_count: int = 0
    skipped: bool = False
    skip_reason: str = ""
    routed_model: str = ""
    streamed: bool = False
    ttft_ms: float | None = None
    cache_hit: bool = False
    highest_chunk_score: float | None = None
    lowest_chunk_score: float | None = None


# Global schema
class RagState(BaseModel):
    model_config = ConfigDict(extra="forbid", validate_assignment=True)

    session_id: str
    user_query: str
    phase: str = "planner"
    chat_history: list[Message] = Field(default_factory=list)
    history: list[HistoryTurn] = Field(default_factory=list)
    planner_state: PlannerState = Field(default_factory=PlannerState)
    fetcher_state: FetcherState = Field(default_factory=FetcherState)
    thinker_state: ThinkerState = Field(default_factory=ThinkerState)
    thinker_timing: ThinkerTiming = Field(default_factory=ThinkerTiming)
    debug_trace: list[str] = Field(default_factory=list)
    node_latencies_ms: dict[str, float] = Field(default_factory=dict)
    created_at: str = Field(default_factory=now_utc_iso)

    def add_trace(self, message: str) -> None:
        self.debug_trace.append(f"{now_utc_iso()} | {message}")
