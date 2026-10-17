from __future__ import annotations

import hashlib
import json
import logging
import time
from typing import Iterable, Iterator

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI

from core.env import settings
from core.rag.prompt import (
    MODEL_DICT,
    THINKER_JSON_INSTRUCTIONS,
    THINKER_PROMPT,
    THINKER_PROMPT_VERSION,
)
from core.rag.schema import ChunkHit, RagState, ThinkerState, ThinkerTiming


LOGGER = logging.getLogger(__name__)


class ThinkerNode:
    def __init__(self) -> None:
        self.llm = None
        self.structured_llm = None
        self.fast_llm = None
        self.fast_structured_llm = None
        self._cache = None
        if settings.OPENAI_API_KEY:
            self.llm = ChatOpenAI(
                model=MODEL_DICT["THINKER"],
                api_key=settings.OPENAI_API_KEY,
                base_url=settings.OPENAI_BASE_URL or None,
                temperature=0.1,
            )
            self.structured_llm = self.llm.with_structured_output(ThinkerState)
            if settings.THINKER_FAST_ENABLED:
                self.fast_llm = ChatOpenAI(
                    model=settings.THINKER_FAST_MODEL,
                    api_key=settings.OPENAI_API_KEY,
                    base_url=settings.OPENAI_BASE_URL or None,
                    temperature=0.1,
                )
                self.fast_structured_llm = self.fast_llm.with_structured_output(
                    ThinkerState
                )
        if settings.THINKER_CACHE_ENABLED:
            try:
                from core.rag.thinker_cache import ThinkerCache

                self._cache = ThinkerCache()
            except Exception as exc:
                LOGGER.warning("Thinker cache unavailable: %s", exc)
                self._cache = None

    def run(self, state: RagState, *, on_delta=None) -> RagState:
        """Run the Thinker.

        ``on_delta`` (optional) is called with each streamed answer-text
        fragment when the LLM path streams; it is never called for cached,
        skipped, or fallback results.
        """
        timing = ThinkerTiming()
        start = time.perf_counter()

        skip_reason = self._skip_reason(state)
        if skip_reason:
            thinker_state = self._run_skipped(state, timing, skip_reason)
        else:
            thinker_state = self._run_cached_or_llm(state, timing, on_delta)

        thinker_state = self._postprocess_thinker_state(
            thinker_state, state.fetcher_state.chunks
        )

        timing.latency_ms = round((time.perf_counter() - start) * 1000, 2)
        timing.used_chunk_count = len(thinker_state.used_chunk_ids)
        timing.output_chars = len(thinker_state.response)
        timing.estimated_output_tokens = self._estimate_tokens(thinker_state.response)
        chunk_scores = [c.score for c in state.fetcher_state.chunks]
        if chunk_scores:
            timing.highest_chunk_score = max(chunk_scores)
            timing.lowest_chunk_score = min(chunk_scores)

        state.thinker_state = thinker_state
        state.thinker_timing = timing
        state.add_trace(
            f"Thinker node completed: sufficient={thinker_state.sufficient}, "
            f"used_chunks={len(thinker_state.used_chunk_ids)}, "
            f"latency_ms={timing.latency_ms}, "
            f"input_chunks={timing.input_chunk_count}, "
            f"input_chars={timing.input_chars}, "
            f"est_input_tokens={timing.estimated_input_tokens}, "
            f"output_chars={timing.output_chars}, "
            f"est_output_tokens={timing.estimated_output_tokens}"
        )
        return state

    def _run_skipped(
        self, state: RagState, timing: ThinkerTiming, skip_reason: str
    ) -> ThinkerState:
        """Deterministic fast path for low-confidence retrieval (no LLM call)."""
        timing.skipped = True
        timing.skip_reason = skip_reason
        state.add_trace(f"Thinker skipped LLM: {skip_reason}.")
        return self._fallback_thinker(state)

    def _run_cached_or_llm(
        self, state: RagState, timing: ThinkerTiming, on_delta
    ) -> ThinkerState:
        """Serve from the response cache when possible, else via model routing."""
        cache_key = self._cache_key(state)
        cached = self._cache_get(cache_key)
        if cached is not None:
            timing.cache_hit = True
            state.add_trace("Thinker cache hit.")
            return cached

        thinker_state = self._run_routed_llm(state, timing, on_delta)
        self._cache_put(cache_key, thinker_state)
        return thinker_state

    def _run_routed_llm(
        self, state: RagState, timing: ThinkerTiming, on_delta
    ) -> ThinkerState:
        """Route to the fast or full model by query difficulty.

        Only the full model path streams: the fast model is invoked without
        ``on_delta`` because its answers are short and not streamed.
        """
        use_fast = self._use_fast_model(state)
        timing.routed_model = (
            settings.THINKER_FAST_MODEL if use_fast else MODEL_DICT["THINKER"]
        )
        if use_fast and self.fast_structured_llm:
            return self._run_llm_thinker(
                state, timing, structured_llm=self.fast_structured_llm
            )
        if self.structured_llm:
            return self._run_llm_thinker(state, timing, on_delta=on_delta)
        return self._fallback_thinker(state)

    @staticmethod
    def _estimate_tokens(text: str) -> int:
        """Rough token estimate (~4 chars per token for English text)."""
        return max(1, len(text) // 4) if text else 0

    def _run_llm_thinker(
        self,
        state: RagState,
        timing: ThinkerTiming,
        *,
        structured_llm=None,
        on_delta=None,
    ) -> ThinkerState:
        structured_llm = structured_llm or self.structured_llm
        if structured_llm is None:
            raise RuntimeError("No structured Thinker LLM is available.")
        chunks_payload = self._build_chunks_payload(state.fetcher_state.chunks, timing)
        user_payload = {
            "user_query": state.user_query,
            "chunks": chunks_payload,
        }
        user_prompt = self._build_user_thinker_prompt(user_payload)
        timing.input_chars = len(THINKER_PROMPT) + len(user_prompt)
        timing.estimated_input_tokens = self._estimate_tokens(
            THINKER_PROMPT
        ) + self._estimate_tokens(user_prompt)
        if on_delta is not None:
            return self._run_llm_thinker_streaming(state, timing, user_prompt, on_delta)
        try:
            parsed = structured_llm.invoke(
                [
                    SystemMessage(content=THINKER_PROMPT),
                    HumanMessage(content=user_prompt),
                ]
            )
            return parsed
        except Exception as exc:
            LOGGER.warning("Thinker LLM failed, using heuristic fallback: %s", exc)
            state.add_trace(f"Thinker LLM failed: {exc}. Using fallback.")
            return self._fallback_thinker(state)

    def _run_llm_thinker_streaming(
        self,
        state: RagState,
        timing: ThinkerTiming,
        user_prompt: str,
        on_delta,
    ) -> ThinkerState:
        """Stream raw JSON text from the LLM, emitting answer deltas.

        The model is called with ``response_format={"type": "json_object"}``
        and THINKER_JSON_INSTRUCTIONS appended to the system prompt. Deltas are
        forwarded via ``on_delta`` as they arrive; the accumulated JSON is
        parsed and validated into ``ThinkerState`` only after the stream ends,
        so final validation always happens before the completed answer.
        """
        if self.llm is None:
            raise RuntimeError("Streaming requires the full Thinker LLM.")
        messages = [
            SystemMessage(content=THINKER_PROMPT + THINKER_JSON_INSTRUCTIONS),
            HumanMessage(content=user_prompt),
        ]
        start = time.perf_counter()
        parts: list[str] = []
        try:
            stream = self.llm.stream(
                messages,
                response_format={"type": "json_object"},
            )
            for chunk in stream:
                text = getattr(chunk, "content", None)
                if not text or not isinstance(text, str):
                    continue
                if not parts:
                    timing.ttft_ms = round((time.perf_counter() - start) * 1000, 2)
                parts.append(text)
                on_delta(text)
        except Exception as exc:
            LOGGER.warning(
                "Thinker streaming failed, using heuristic fallback: %s", exc
            )
            state.add_trace(f"Thinker streaming failed: {exc}. Using fallback.")
            return self._fallback_thinker(state)

        timing.streamed = True
        raw = "".join(parts).strip()
        try:
            parsed = ThinkerState.model_validate_json(raw)
            return parsed
        except Exception as exc:
            LOGGER.warning("Thinker streamed JSON invalid, using fallback: %s", exc)
            state.add_trace(f"Thinker streamed JSON invalid: {exc}. Using fallback.")
            return self._fallback_thinker(state)

    def _cache_key(self, state: RagState) -> str:
        """Deterministic cache key over query, context, model, and config."""
        chunk_ids = [c.chunk_id for c in state.fetcher_state.chunks]
        fingerprint = "|".join(
            [
                state.user_query.strip().lower(),
                ",".join(chunk_ids),
                MODEL_DICT["THINKER"],
                settings.THINKER_FAST_MODEL,
                THINKER_PROMPT_VERSION,
                str(settings.THINKER_MAX_CHUNKS),
                str(settings.THINKER_FULL_TEXT_CHUNKS),
                str(settings.THINKER_MAX_CHUNK_CHARS),
                str(settings.THINKER_CONTEXT_BUDGET),
            ]
        )
        return hashlib.sha256(fingerprint.encode("utf-8")).hexdigest()

    def _cache_get(self, key: str) -> ThinkerState | None:
        if self._cache is None:
            return None
        try:
            return self._cache.get(key)
        except Exception as exc:
            LOGGER.warning("Thinker cache get failed: %s", exc)
            return None

    def _cache_put(self, key: str, thinker_state: ThinkerState) -> None:
        if self._cache is None:
            return
        try:
            self._cache.put(key, thinker_state)
        except Exception as exc:
            LOGGER.warning("Thinker cache put failed: %s", exc)

    def _build_chunks_payload(
        self, chunks: list[ChunkHit], timing: ThinkerTiming
    ) -> list[dict]:
        """Build the Thinker context payload under an adaptive budget.

        Fetcher order is assumed to be ranked best-first. Top
        ``THINKER_FULL_TEXT_CHUNKS`` keep full text; lower-ranked chunks are
        truncated to ``THINKER_MAX_CHUNK_CHARS``, then from lowest-ranked
        upward until the ``THINKER_CONTEXT_BUDGET`` total is met.
        """
        # Guards against a misconfigured negative full-text setting.
        full_text_count = max(0, settings.THINKER_FULL_TEXT_CHUNKS)
        ranked_chunks = list(chunks)[: settings.THINKER_MAX_CHUNKS]
        chunk_texts = [c.text for c in ranked_chunks]

        # Per-chunk truncation: full text for top-ranked, shortened for the rest.
        for i in range(len(chunk_texts)):
            if (
                i >= full_text_count
                and len(chunk_texts[i]) > settings.THINKER_MAX_CHUNK_CHARS
            ):
                chunk_texts[i] = (
                    chunk_texts[i][: settings.THINKER_MAX_CHUNK_CHARS].rstrip() + "…"
                )

        # Global budget: truncate from the lowest-ranked chunk upward.
        total_chars = sum(len(text) for text in chunk_texts)
        lowest_ranked = len(chunk_texts) - 1
        while (
            total_chars > settings.THINKER_CONTEXT_BUDGET
            and lowest_ranked >= full_text_count
        ):
            excess = total_chars - settings.THINKER_CONTEXT_BUDGET
            cut = min(excess, len(chunk_texts[lowest_ranked]) - 1)
            if cut > 0:
                chunk_texts[lowest_ranked] = (
                    chunk_texts[lowest_ranked][:cut].rstrip() + "…"
                )
                total_chars -= cut
            lowest_ranked -= 1

        timing.input_chunk_count = len(ranked_chunks)
        timing.dropped_chunk_count = len(chunks) - len(ranked_chunks)
        timing.truncated_chunk_count = sum(
            1
            for chunk, text in zip(ranked_chunks, chunk_texts)
            if len(text) < len(chunk.text)
        )
        return [
            {
                "chunk_id": chunk.chunk_id,
                "page_id": chunk.page_id,
                "page_title": chunk.page_title,
                "section": chunk.section,
                "subsection": chunk.subsection,
                "score": chunk.score,
                "text": text,
            }
            for chunk, text in zip(ranked_chunks, chunk_texts)
        ]

    def _skip_reason(self, state: RagState) -> str:
        """Return a reason string when the Thinker LLM can be skipped, else empty."""
        if not settings.THINKER_SKIP_ENABLED:
            return ""
        chunks = state.fetcher_state.chunks
        if not chunks:
            return "no_chunks"
        scores = [c.score for c in chunks]
        top = max(scores)
        if top < settings.THINKER_SKIP_MIN_SCORE:
            return f"top_score_below_min ({top:.4f})"
        rerank_failed = all(
            timing.rerank_passed is False
            for timing in state.fetcher_state.query_timings
            if timing.rerank_passed is not None
        )
        has_rerank_metadata = any(
            timing.rerank_passed is not None
            for timing in state.fetcher_state.query_timings
        )
        if (
            top < settings.THINKER_SKIP_MAX_SCORE
            and len(chunks) <= 1
            and has_rerank_metadata
            and rerank_failed
        ):
            return f"low_score_sparse_retrieval ({top:.4f})"
        return ""

    def _use_fast_model(self, state: RagState) -> bool:
        """Route to the fast model for simple, high-confidence queries.

        Uses only existing planner metadata and retrieval signals — no extra
        LLM classifier call.
        """
        if not settings.THINKER_FAST_ENABLED or not self.fast_structured_llm:
            return False
        chunks = state.fetcher_state.chunks
        if not chunks:
            return False
        # Short, direct question.
        if len(state.user_query) > settings.THINKER_FAST_MAX_QUERY_CHARS:
            return False
        # High-confidence retrieval: strong top score.
        top_score = max(c.score for c in chunks)
        if top_score < settings.THINKER_FAST_MIN_TOP_SCORE:
            return False
        # Single-topic: few strong chunks, no multi-part planner expansion.
        if len(chunks) > settings.THINKER_FAST_MAX_CHUNKS:
            return False
        if len(state.planner_state.planned_queries) > 1:
            return False
        return True

    def _fallback_thinker(self, state: RagState) -> ThinkerState:
        threshold = 0.4
        used_ids: list[str] = []
        for c in state.fetcher_state.chunks:
            if c.score <= threshold:
                used_ids.append(c.chunk_id)

        sufficient = bool(used_ids)

        return ThinkerState(
            sufficient=sufficient,
            used_chunk_ids=used_ids,
            response="",
            reasoning=[],
        )

    def _build_user_thinker_prompt(self, payload: dict) -> str:
        return json.dumps(payload, ensure_ascii=False)

    def _postprocess_thinker_state(
        self, thinker_state: ThinkerState, chunks: Iterable[ChunkHit]
    ) -> ThinkerState:
        available_ids = {c.chunk_id for c in chunks}
        thinker_state.used_chunk_ids = [
            cid for cid in thinker_state.used_chunk_ids if cid in available_ids
        ]

        if thinker_state.sufficient and not thinker_state.used_chunk_ids:
            thinker_state.sufficient = False

        if not thinker_state.sufficient:
            if not thinker_state.response.strip():
                if thinker_state.used_chunk_ids:
                    thinker_state.response = (
                        "I found some potentially related passages, but they are not strong enough to answer your question confidently. "
                        "What platform/version are you playing, and what does “easiest” mean to you (combat, bosses, or progression)?"
                    )
                else:
                    thinker_state.response = (
                        "I could not find relevant context for your request in the retrieved chunks. "
                        "Can you clarify the exact game title/version and what you mean by “easiest”?"
                    )
            thinker_state.response = self._append_references_if_needed(
                response=thinker_state.response,
                used_chunk_ids=thinker_state.used_chunk_ids,
                chunks=chunks,
            )
            return thinker_state

        if not thinker_state.response.strip():
            thinker_state.response = (
                "I have enough context to answer, but the response text was empty."
            )

        thinker_state.response = self._append_references_if_needed(
            response=thinker_state.response,
            used_chunk_ids=thinker_state.used_chunk_ids,
            chunks=chunks,
        )
        return thinker_state

    def _append_references_if_needed(
        self,
        *,
        response: str,
        used_chunk_ids: list[str],
        chunks: Iterable[ChunkHit],
    ) -> str:
        if not used_chunk_ids:
            return response

        # References already present means a cache hit or a retried answer.
        if "isu.fandom.com/wiki/?curid=" in response:
            return response

        chunk_by_id = {c.chunk_id: c for c in chunks}
        page_ids: list[str] = []
        seen: set[str] = set()
        for chunk_id in used_chunk_ids:
            hit = chunk_by_id.get(chunk_id)
            if not hit:
                continue
            page_id = str(hit.page_id).strip()
            if not page_id or page_id in seen:
                continue
            seen.add(page_id)
            page_ids.append(page_id)

        if not page_ids:
            return response

        refs = "\n".join(
            f"https://isu.fandom.com/wiki/?curid={pid}" for pid in page_ids
        )
        return f"{response.rstrip()}\n\nReferences:\n{refs}"
