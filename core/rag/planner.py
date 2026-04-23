from __future__ import annotations

import json
import logging
import re

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI

from core.env import settings
from core.rag.prompt import PLANNER_PROMPT
from core.rag.schema import PlannerState, RagState


LOGGER = logging.getLogger(__name__)


class PlannerNode:
    def __init__(self) -> None:
        self.llm = None
        self.structured_llm = None
        if settings.PLANNER_AGENT and settings.OPENAI_API_KEY:
            self.llm = ChatOpenAI(
                model=settings.PLANNER_MODEL,
                api_key=settings.OPENAI_API_KEY,
                base_url=settings.OPENAI_BASE_URL or None,
                temperature=0.1,
            )
            self.structured_llm = self.llm.with_structured_output(PlannerState)

    def run(self, state: RagState) -> RagState:
        output = (
            self._run_llm_planner(state) if self.llm else self._fallback_planner(state)
        )
        state.planner_state = output
        state.add_trace(f"Planner node completed with source={output.source}.")
        return state

    def _run_llm_planner(self, state: RagState) -> PlannerState:
        if not self.structured_llm:
            state.add_trace("Planner LLM skipped: not initialized. Using fallback.")
            return self._fallback_planner(state)

        history_hint = [
            {"q": h.q, "a": h.a} for h in state.history[-settings.RAG_HISTORY_WINDOW :]
        ]
        user_payload = {
            "user_query": state.user_query,
            "history": history_hint,
        }
        try:
            parsed = self.structured_llm.invoke(
                [
                    SystemMessage(content=PLANNER_PROMPT),
                    HumanMessage(content=json.dumps(user_payload, ensure_ascii=False)),
                ]
            )
            condensed_query = str(getattr(parsed, "condensed_query", "")).strip()
            if not condensed_query:
                condensed_query = state.user_query.strip()

            entities: list[str] = []
            seen_entities: set[str] = set()
            for item in list(getattr(parsed, "entities", []) or []):
                text = str(item).strip()
                if not text:
                    continue
                key = text.lower()
                if key in seen_entities:
                    continue
                seen_entities.add(key)
                entities.append(text)

            planned_queries: list[str] = []
            seen: set[str] = set()
            for candidate in parsed.planned_queries:
                text = str(candidate).strip()
                if not text:
                    continue
                if not settings.PLANNER_PRESERVE_RICH_QUERY:
                    text = self._collapse_to_noun_phrase(text)
                if not text:
                    continue
                key = text.lower()
                if key in seen:
                    continue
                seen.add(key)
                planned_queries.append(text)

            if not planned_queries and condensed_query:
                fallback_query = condensed_query
                if not settings.PLANNER_PRESERVE_RICH_QUERY:
                    fallback_query = self._collapse_to_noun_phrase(fallback_query)
                if fallback_query:
                    planned_queries = [fallback_query]
            if not planned_queries:
                raise ValueError("Planner LLM returned empty planned_queries.")

            reasoning = parsed.reasoning or [
                "Generated semantic search query variants from user intent."
            ]
            return PlannerState(
                condensed_query=condensed_query or planned_queries[0],
                planned_queries=planned_queries,
                entities=entities[:12],
                reasoning=[str(item) for item in reasoning][:5],
                source="llm",
            )
        except Exception as exc:
            LOGGER.warning("Planner LLM failed, using fallback planner: %s", exc)
            state.add_trace(f"Planner LLM failed: {exc}. Using fallback.")
            return self._fallback_planner(state)

    def _fallback_planner(self, state: RagState) -> PlannerState:
        fallback_query = state.user_query.strip()
        if not settings.PLANNER_PRESERVE_RICH_QUERY:
            fallback_query = self._collapse_to_noun_phrase(fallback_query)
        return PlannerState(
            condensed_query=fallback_query,
            planned_queries=[fallback_query] if fallback_query else [],
            entities=[],
            reasoning=[
                "Fell back to original user query due to unavailable planner LLM output.",
            ],
            source="fallback_python",
        )

    def _collapse_to_noun_phrase(self, query: str) -> str:
        text = query.strip()
        patterns = [
            r"^\s*who\s+is\s+",
            r"^\s*what\s+is\s+",
            r"^\s*what\s+are\s+",
            r"^\s*tell\s+me\s+about\s+",
            r"^\s*can\s+you\s+explain\s+",
            r"^\s*i\s+want\s+to\s+know\s+about\s+",
        ]
        for pattern in patterns:
            text = re.sub(pattern, "", text, flags=re.IGNORECASE)
        text = re.sub(r"\s+", " ", text).strip(" ?!.")
        return text or query.strip()
