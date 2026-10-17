from __future__ import annotations

import warnings
import logging
import queue
import threading
from collections.abc import Iterator
from typing import Any

from core.env import settings
from core.rag.history import RedisHistoryStore, generate_session_id
from core.rag.rag_graph import RagGraph
from core.rag.schema import RagState
from core.api.schemas import RagStreamEvent, build_chat_response
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

    # ---- lifecycle -----------------------------------------------------------

    def startup(self) -> None:
        """Called once when the application starts (FastAPI lifespan)."""
        self._vector_store = create_vector_store()
        self._init_history_store()
        self._init_rag()

    def shutdown(self) -> None:
        """Called once when the application shuts down."""
        self._rag = None
        self._history_store = None
        self._vector_store = None

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

    def stream_rag(
        self, query: str, session_id: str | None = None
    ) -> Iterator[RagStreamEvent]:
        """Run the pipeline and yield phase-progress events."""
        sid = (
            str(session_id).strip()
            if session_id and str(session_id).strip()
            else generate_session_id()
        )
        state = RagState(session_id=sid, user_query=query)
        yield RagStreamEvent(
            event="pipeline_started",
            data={"session_id": sid, "phase": "pipeline"},
        )

        if self._history_store is not None:
            try:
                state.history = self._history_store.load_history(sid)
                state.add_trace(f"Loaded session history turns={len(state.history)}.")
            except Exception as exc:
                LOGGER.warning("Failed to load session history: %s", exc)
                state.add_trace(f"Failed to load session history: {exc}")

        try:
            rag = self._ensure_rag()
            phase_by_node = {
                "planner": "planner",
                "fetcher": "retrieval",
                "thinker": "generation",
                "context": "context",
            }

            yield RagStreamEvent(
                event="planner_started",
                data={"session_id": sid, "phase": "planner"},
            )

            # Run the pipeline in a background thread so Thinker answer
            # deltas can be yielded to the client as they arrive (real
            # streaming, not buffered per-node). The queue carries:
            #   ("answer_delta", text)     — streamed Thinker output fragment
            #   ("node", node_name, state) — a graph node completed
            #   ("error", exc)             — pipeline failure
            #   None                        — graph finished
            events: queue.Queue = queue.Queue()

            def _on_delta(text: str) -> None:
                events.put(("answer_delta", text))

            def _run_graph() -> None:
                try:
                    for node_name, node_state in rag.stream(state, on_delta=_on_delta):
                        events.put(("node", node_name, node_state))
                except Exception as exc:  # noqa: BLE001 - forwarded to consumer
                    events.put(("error", exc))
                finally:
                    events.put(None)

            worker = threading.Thread(target=_run_graph, daemon=True)
            worker.start()

            while True:
                item = events.get()
                if item is None:
                    break
                kind = item[0]
                if kind == "answer_delta":
                    yield RagStreamEvent(
                        event="answer_delta",
                        data={"session_id": sid, "text": item[1]},
                    )
                    continue
                if kind == "error":
                    raise item[1]

                node_name, state = item[1], item[2]
                phase = phase_by_node.get(node_name, node_name)
                completed_data: dict[str, Any] = {
                    "session_id": sid,
                    "phase": phase,
                }
                latency = state.node_latencies_ms.get(node_name)
                if latency is not None:
                    completed_data["latency_ms"] = latency
                if node_name == "planner":
                    completed_data["planned_query_count"] = len(
                        state.planner_state.planned_queries
                    )
                    completed_data["source"] = state.planner_state.source
                elif node_name == "fetcher":
                    completed_data["chunk_count"] = len(state.fetcher_state.chunks)

                yield RagStreamEvent(
                    event=f"{phase}_completed",
                    data=completed_data,
                )

                # The next "*_started" event is emitted here, i.e. before the
                # graph advances to the following node on the next iteration.
                if node_name == "planner":
                    yield RagStreamEvent(
                        event="retrieval_started",
                        data={"session_id": sid, "phase": "retrieval"},
                    )
                elif node_name == "fetcher":
                    yield RagStreamEvent(
                        event="generation_started",
                        data={"session_id": sid, "phase": "generation"},
                    )
                elif node_name == "thinker":
                    yield RagStreamEvent(
                        event="context_started",
                        data={"session_id": sid, "phase": "context"},
                    )

            if self._history_store is not None:
                response_text = (state.thinker_state.response or "").strip()
                question_text = (state.user_query or "").strip()
                if question_text and response_text:
                    try:
                        self._history_store.append_turn(
                            state.session_id,
                            question=question_text,
                            answer=response_text,
                            history_window=settings.RAG_HISTORY_WINDOW,
                            ttl_seconds=settings.RAG_SESSION_TTL_SECONDS,
                        )
                        state.history = self._history_store.load_history(
                            state.session_id
                        )
                        state.add_trace(
                            f"Persisted session history turns={len(state.history)}."
                        )
                    except Exception as exc:
                        LOGGER.warning("Failed to persist session history: %s", exc)
                        state.add_trace(f"Failed to persist session history: {exc}")

            yield RagStreamEvent(
                event="final_answer",
                data=build_chat_response(state).model_dump(),
            )
            yield RagStreamEvent(
                event="pipeline_completed",
                data={"session_id": sid, "phase": "pipeline"},
            )
        except Exception as exc:
            LOGGER.exception("Streaming RAG pipeline failed")
            yield RagStreamEvent(
                event="pipeline_error",
                data={
                    "session_id": sid,
                    "error_type": type(exc).__name__,
                    "detail": str(exc),
                },
            )

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
