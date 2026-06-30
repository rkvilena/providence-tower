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

    EMBEDDING_MODEL: str

    OPENAI_API_KEY: str | None
    OPENAI_BASE_URL: str | None

    REDIS_HOST: str
    REDIS_PORT: int
    REDIS_DB: int
    REDIS_PASSWORD: str | None

    RAG_HISTORY_WINDOW: int
    RAG_SESSION_TTL_SECONDS: int

    PLANNER_AGENT: bool
    PLANNER_MODEL: str
    PLANNER_PRESERVE_RICH_QUERY: bool

    RERANK_ENABLED: bool
    RERANK_MODEL: str
    RERANK_TOP_K: int
    RERANK_MIN_TOP_SCORE: float

    RATE_LIMIT_SHORT_WINDOW_SECONDS: int
    RATE_LIMIT_SHORT_MAX_REQUESTS: int
    RATE_LIMIT_LONG_WINDOW_SECONDS: int
    RATE_LIMIT_LONG_MAX_REQUESTS: int

    VECTOR_STORE_PROVIDER: str
    UPSTASH_REDIS_URL: str | None
    UPSTASH_VECTOR_URL: str | None
    UPSTASH_VECTOR_TOKEN: str | None


settings = Settings(
    EMBEDDING_MODEL=os.getenv("EMBEDDING_MODEL", "BAAI/bge-small-en-v1.5"),
    OPENAI_API_KEY=os.getenv("OPENAI_API_KEY"),
    OPENAI_BASE_URL=os.getenv("OPENAI_BASE_URL"),
    REDIS_HOST=os.getenv("REDIS_HOST", "127.0.0.1"),
    REDIS_PORT=_to_int(os.getenv("REDIS_PORT"), default=6379),
    REDIS_DB=_to_int(os.getenv("REDIS_DB"), default=0),
    REDIS_PASSWORD=os.getenv("REDIS_PASSWORD"),
    RAG_HISTORY_WINDOW=_to_int(os.getenv("RAG_HISTORY_WINDOW"), default=10),
    RAG_SESSION_TTL_SECONDS=_to_int(os.getenv("RAG_SESSION_TTL_SECONDS"), default=900),
    PLANNER_AGENT=_to_bool(os.getenv("PLANNER_AGENT"), default=True),
    PLANNER_MODEL=os.getenv("PLANNER_MODEL", MODEL_DICT["PLANNER"]),
    PLANNER_PRESERVE_RICH_QUERY=_to_bool(
        os.getenv("PLANNER_PRESERVE_RICH_QUERY"), default=False
    ),
    RERANK_ENABLED=_to_bool(os.getenv("RERANK_ENABLED"), default=True),
    RERANK_MODEL=os.getenv("RERANK_MODEL", "cross-encoder/ms-marco-MiniLM-L-6-v2"),
    RERANK_TOP_K=_to_int(os.getenv("RERANK_TOP_K"), default=20),
    RERANK_MIN_TOP_SCORE=_to_float(os.getenv("RERANK_MIN_TOP_SCORE"), default=0.7),
    RATE_LIMIT_SHORT_WINDOW_SECONDS=_to_int(
        os.getenv("RATE_LIMIT_SHORT_WINDOW_SECONDS"), default=60
    ),
    RATE_LIMIT_SHORT_MAX_REQUESTS=_to_int(
        os.getenv("RATE_LIMIT_SHORT_MAX_REQUESTS"), default=5
    ),
    RATE_LIMIT_LONG_WINDOW_SECONDS=_to_int(
        os.getenv("RATE_LIMIT_LONG_WINDOW_SECONDS"), default=86400
    ),
    RATE_LIMIT_LONG_MAX_REQUESTS=_to_int(
        os.getenv("RATE_LIMIT_LONG_MAX_REQUESTS"), default=30
    ),
    VECTOR_STORE_PROVIDER=os.getenv("VECTOR_STORE_PROVIDER", "upstash"),
    UPSTASH_REDIS_URL=os.getenv("UPSTASH_REDIS_URL"),
    UPSTASH_VECTOR_URL=os.getenv("UPSTASH_VECTOR_URL"),
    UPSTASH_VECTOR_TOKEN=os.getenv("UPSTASH_VECTOR_TOKEN"),
)
