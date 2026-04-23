from __future__ import annotations

import json

import logging
from typing import Iterable

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI

from core.env import settings
from core.rag.prompt import MODEL_DICT, THINKER_PROMPT
from core.rag.schema import ChunkHit, RagState, ThinkerState


LOGGER = logging.getLogger(__name__)


class ThinkerNode:
    def __init__(self) -> None:
        self.llm = None
        self.structured_llm = None
        if settings.OPENAI_API_KEY:
            self.llm = ChatOpenAI(
                model=MODEL_DICT["THINKER"],
                api_key=settings.OPENAI_API_KEY,
                base_url=settings.OPENAI_BASE_URL or None,
                temperature=0.1,
            )
            self.structured_llm = self.llm.with_structured_output(ThinkerState)

    def run(self, state: RagState) -> RagState:
        if self.structured_llm:
            thinker_state = self._run_llm_thinker(state)
        else:
            thinker_state = self._fallback_thinker(state)

        thinker_state = self._postprocess_thinker_state(
            thinker_state, state.fetcher_state.chunks
        )

        state.thinker_state = thinker_state
        state.add_trace(
            f"Thinker node completed: sufficient={thinker_state.sufficient}, "
            f"used_chunks={len(thinker_state.used_chunk_ids)}"
        )
        return state

    def _run_llm_thinker(self, state: RagState) -> ThinkerState:
        assert self.structured_llm is not None
        chunks_payload = [
            {
                "chunk_id": c.chunk_id,
                "page_id": c.page_id,
                "page_title": c.page_title,
                "section": c.section,
                "subsection": c.subsection,
                "score": c.score,
                "text": c.text,
            }
            for c in state.fetcher_state.chunks
        ]
        user_payload = {
            "user_query": state.user_query,
            "chunks": chunks_payload,
        }
        try:
            parsed = self.structured_llm.invoke(
                [
                    SystemMessage(content=THINKER_PROMPT),
                    HumanMessage(content=self._build_user_thinker_prompt(user_payload)),
                ]
            )
            return parsed
        except Exception as exc:
            LOGGER.warning("Thinker LLM failed, using heuristic fallback: %s", exc)
            state.add_trace(f"Thinker LLM failed: {exc}. Using fallback.")
            return self._fallback_thinker(state)

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
            reasoning=[
                "Heuristic fallback based on score threshold.",
            ],
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

        existing = response
        if "isu.fandom.com/wiki/?curid=" in existing:
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
