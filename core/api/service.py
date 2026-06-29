from __future__ import annotations

import warnings
import logging
from typing import Any

from core.env import settings
from core.rag.history import RedisHistoryStore, generate_session_id
from core.rag.rag_graph import RagGraph
from core.rag.schema import RagState
from core.vector_store.protocol import VectorStoreProtocol
from core.vector_store_factory import create_vector_store

LOGGER = logging.getLogger(__name__)


class RagService:
    """Singleton-style service that wraps RagGraph + session history management."""

    def __init__(self) -> None:
        # Suppress Hugging Face Hub deprecation warnings
        warnings.filterwarnings(
            "ignore",
            message=".*resume_download.*is deprecated.*",
            category=FutureWarning,
            module="huggingface_hub",
        )

        self._rag: RagGraph | None = None
        self._history_store: RedisHistoryStore | None = None
        self._vector_store: VectorStoreProtocol | None = None
        self._warmed_up: bool = False

    # ---- lifecycle -----------------------------------------------------------

    def startup(self) -> None:
        """Called once when the application starts (FastAPI lifespan)."""
        self._vector_store = create_vector_store()
        self._init_history_store()
        self._init_rag()
        self._warmup()

    def shutdown(self) -> None:
        """Called once when the application shuts down."""
        self._rag = None
        self._history_store = None
        self._vector_store = None
        self._warmed_up = False

    # ---- public helpers ------------------------------------------------------

    @property
    def history_store(self) -> RedisHistoryStore | None:
        return self._history_store

    @property
    def is_history_available(self) -> bool:
        return self._history_store is not None

    @property
    def is_planner_llm_enabled(self) -> bool:
        return bool(settings.OPENAI_API_KEY and settings.PLANNER_AGENT)

    @property
    def is_reranker_enabled(self) -> bool:
        return bool(settings.RERANK_ENABLED)

    # ---- core operation ------------------------------------------------------

    def run_rag(self, query: str, session_id: str | None = None) -> RagState:
        """Execute the full RAG pipeline and return the final state.

        Parameters
        ----------
        query:
            The user query text.
        session_id:
            An existing session ID.  If *None* a new session is generated.

        Returns
        -------
        The final ``RagState`` after the graph has run.
        """
        if session_id and str(session_id).strip():
            sid = str(session_id).strip()
        else:
            sid = generate_session_id()

        state = RagState(session_id=sid, user_query=query)

        # Load history for an existing session
        if self._history_store is not None:
            try:
                state.history = self._history_store.load_history(sid)
                state.add_trace(f"Loaded session history turns={len(state.history)}.")
            except Exception as exc:
                LOGGER.warning("Failed to load session history: %s", exc)
                state.add_trace(f"Failed to load session history: {exc}")

        rag = self._ensure_rag()
        result = rag.run(state)

        # Persist Q&A turn after successful completion
        if self._history_store is not None:
            response_text = (result.thinker_state.response or "").strip()
            question_text = (result.user_query or "").strip()
            if question_text and response_text:
                try:
                    self._history_store.append_turn(
                        result.session_id,
                        question=question_text,
                        answer=response_text,
                        history_window=settings.RAG_HISTORY_WINDOW,
                        ttl_seconds=settings.RAG_SESSION_TTL_SECONDS,
                    )
                    result.history = self._history_store.load_history(result.session_id)
                    result.add_trace(
                        f"Persisted session history turns={len(result.history)}."
                    )
                except Exception as exc:
                    LOGGER.warning("Failed to persist session history: %s", exc)
                    result.add_trace(f"Failed to persist session history: {exc}")

        return result

    def clear_session(self, session_id: str) -> bool:
        """Clear history for a given session.

        Returns *True* on success (or when Redis is unavailable).
        """
        if self._history_store is None:
            return True
        try:
            self._history_store.clear_history(session_id)
            return True
        except Exception as exc:
            LOGGER.warning("Failed to clear session '%s': %s", session_id, exc)
            return False

    # ---- internals -----------------------------------------------------------

    def _init_history_store(self) -> None:
        store = RedisHistoryStore()
        try:
            store.ping()
            self._history_store = store
            LOGGER.info("Redis session history: ON")
        except Exception as exc:
            self._history_store = None
            LOGGER.info("Redis session history: OFF (%s)", exc)

    def _init_rag(self) -> None:
        if self._vector_store is None:
            self._vector_store = create_vector_store()
        self._rag = RagGraph(vector_store=self._vector_store)

    def _ensure_rag(self) -> RagGraph:
        if self._rag is None:
            self._init_rag()
        return self._rag

    def _warmup(self) -> None:
        if self._warmed_up:
            return
        rag = self._ensure_rag()
        try:
            rag.warmup()
            LOGGER.info("RAG warmup complete.")
        except Exception as exc:
            LOGGER.warning("RAG warmup failed (non-fatal): %s", exc)
        self._warmed_up = True
