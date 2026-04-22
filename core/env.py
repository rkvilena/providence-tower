from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv
from pydantic import BaseModel, ConfigDict

from core.rag.prompt import MODEL_DICT


PROJECT_ROOT = Path(__file__).resolve().parents[1]
load_dotenv(PROJECT_ROOT / "core" / ".env")
load_dotenv()


def _to_bool(value: str | None, *, default: bool) -> bool:
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


def _to_int(value: str | None, *, default: int) -> int:
    if value is None:
        return default
    try:
        return int(value)
    except ValueError:
        return default


def _to_float(value: str | None, *, default: float) -> float:
    if value is None:
        return default
    try:
        return float(value)
    except ValueError:
        return default


class Settings(BaseModel):
    model_config = ConfigDict(extra="forbid")

    OPENAI_API_KEY: str | None
    OPENAI_BASE_URL: str | None
    PLANNER_AGENT: bool
    PLANNER_MODEL: str
    PLANNER_PRESERVE_RICH_QUERY: bool
    RERANK_ENABLED: bool
    RERANK_MODEL: str
    RERANK_TOP_K: int
    RERANK_MIN_TOP_SCORE: float


settings = Settings(
    OPENAI_API_KEY=os.getenv("OPENAI_API_KEY"),
    OPENAI_BASE_URL=os.getenv("OPENAI_BASE_URL"),
    PLANNER_AGENT=_to_bool(os.getenv("PLANNER_AGENT"), default=True),
    PLANNER_MODEL=os.getenv("PLANNER_MODEL", MODEL_DICT["PLANNER"]),
    PLANNER_PRESERVE_RICH_QUERY=_to_bool(os.getenv("PLANNER_PRESERVE_RICH_QUERY"), default=False),
    RERANK_ENABLED=_to_bool(os.getenv("RERANK_ENABLED"), default=True),
    RERANK_MODEL=os.getenv("RERANK_MODEL", "cross-encoder/ms-marco-MiniLM-L-6-v2"),
    RERANK_TOP_K=_to_int(os.getenv("RERANK_TOP_K"), default=20),
    RERANK_MIN_TOP_SCORE=_to_float(os.getenv("RERANK_MIN_TOP_SCORE"), default=0.7),
)
