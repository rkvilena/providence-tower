from __future__ import annotations

import time
from typing import TypedDict

from core.env import settings
from core.rag.history import RedisHistoryStore
from core.rag.schema import RagState, GraphState


class ContextState(TypedDict):
    persisted: bool
    history_count: int


class ContextNode:
    def __init__(self) -> None:
        self.store = RedisHistoryStore()

    def run(self, state: RagState) -> RagState:
        """Persist successful Q&A to history and reload updated history."""
        response_text = (state.thinker_state.response or "").strip()
        question_text = (state.user_query or "").strip()

        if not question_text or not response_text:
            state.add_trace("Context: Skipped persistence (empty question or response)")
            return state

        try:
            # Persist the successful Q&A turn
            self.store.append_turn(
                state.session_id,
                question=question_text,
                answer=response_text,
                history_window=settings.RAG_HISTORY_WINDOW,
                ttl_seconds=settings.RAG_SESSION_TTL_SECONDS,
            )

            # Reload the updated history
            updated_history = self.store.load_history(state.session_id)
            state.history = updated_history
            state.add_trace(
                f"Context: Persisted session history (turns={len(updated_history)})"
            )

        except Exception as exc:
            state.add_trace(f"Context: Failed to persist history: {exc}")

        return state
