from __future__ import annotations

from datetime import datetime, timezone
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


def now_utc_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


class Message(BaseModel):
    model_config = ConfigDict(extra="forbid")

    role: str
    content: str
    timestamp: str = Field(default_factory=now_utc_iso)


class PlannerOutput(BaseModel):
    model_config = ConfigDict(extra="forbid")

    planned_query: str
    queries: list[str] = Field(default_factory=list)
    reasoning: list[str] = Field(default_factory=list)
    source: Literal["llm", "fallback_python"] = "fallback_python"


class PlannerLLMResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    primary_query: str = ""
    expansion_queries: list[str] = Field(default_factory=list)
    reasoning: list[str] = Field(default_factory=list)
    needs_expansion: bool = False


class RagState(BaseModel):
    model_config = ConfigDict(extra="forbid", validate_assignment=True)

    session_id: str
    user_query: str
    phase: str = "planner"
    chat_history: list[Message] = Field(default_factory=list)
    planned_query: str = ""
    planned_queries: list[str] = Field(default_factory=list)
    planner_output: PlannerOutput | None = None
    debug_trace: list[str] = Field(default_factory=list)
    node_latencies_ms: dict[str, float] = Field(default_factory=dict)
    created_at: str = Field(default_factory=now_utc_iso)

    def add_trace(self, message: str) -> None:
        self.debug_trace.append(f"{now_utc_iso()} | {message}")
