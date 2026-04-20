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


class Settings(BaseModel):
    model_config = ConfigDict(extra="forbid")

    OPENAI_API_KEY: str | None
    OPENAI_BASE_URL: str | None
    PLANNER_AGENT: bool
    PLANNER_MODEL: str


settings = Settings(
    OPENAI_API_KEY=os.getenv("OPENAI_API_KEY"),
    OPENAI_BASE_URL=os.getenv("OPENAI_BASE_URL"),
    PLANNER_AGENT=_to_bool(os.getenv("PLANNER_AGENT"), default=True),
    PLANNER_MODEL=os.getenv("PLANNER_MODEL", MODEL_DICT["PLANNER"]),
)
