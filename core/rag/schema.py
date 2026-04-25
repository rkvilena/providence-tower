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
    source: Literal["llm", "fallback_python"] = "fallback_python"


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
class FetcherState(BaseModel):
    model_config = ConfigDict(extra="forbid")

    chunks: list[ChunkHit] = Field(default_factory=list)


# Thinker schema
class ThinkerState(BaseModel):
    model_config = ConfigDict(extra="forbid")

    sufficient: bool = False
    used_chunk_ids: list[str] = Field(default_factory=list)
    response: str = ""
    reasoning: list[str] = Field(default_factory=list)


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
    debug_trace: list[str] = Field(default_factory=list)
    node_latencies_ms: dict[str, float] = Field(default_factory=dict)
    created_at: str = Field(default_factory=now_utc_iso)

    def add_trace(self, message: str) -> None:
        self.debug_trace.append(f"{now_utc_iso()} | {message}")
