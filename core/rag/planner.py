from __future__ import annotations

import json
import logging
import re

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI

from core.env import settings
from core.rag.prompt import PLANNER_PROMPT
from core.rag.schema import PlannerLLMResponse, PlannerOutput, RagState


LOGGER = logging.getLogger(__name__)


_STOPWORDS = {
    "a",
    "an",
    "and",
    "are",
    "as",
    "at",
    "be",
    "by",
    "for",
    "from",
    "how",
    "i",
    "in",
    "is",
    "it",
    "of",
    "on",
    "or",
    "please",
    "show",
    "tell",
    "that",
    "the",
    "to",
    "what",
    "when",
    "where",
    "which",
    "who",
    "with",
}


class PlannerNode:
    def run(self, state: RagState) -> RagState:
        output = self._run_llm_planner(state) if settings.PLANNER_AGENT else self._fallback_planner(state)
        state.planned_query = output.planned_query
        state.planned_queries = output.queries
        state.planner_output = output
        state.add_trace(f"Planner node completed with source={output.source}.")
        return state

    def _run_llm_planner(self, state: RagState) -> PlannerOutput:
        if not settings.OPENAI_API_KEY:
            state.add_trace("Planner LLM skipped: OPENAI_API_KEY not set. Using fallback.")
            return self._fallback_planner(state)

        llm = ChatOpenAI(
            model=settings.PLANNER_MODEL or models.PLANNER,
            api_key=settings.OPENAI_API_KEY,
            base_url=settings.OPENAI_BASE_URL or None,
            temperature=0.1,
        )
        structured_llm = llm.with_structured_output(PlannerLLMResponse)
        history_hint = [{"role": m.role, "content": m.content} for m in state.chat_history[-4:]]
        user_payload = {
            "user_query": state.user_query,
            "chat_history": history_hint,
        }
        try:
            parsed = structured_llm.invoke(
                [
                    SystemMessage(content=PLANNER_PROMPT),
                    HumanMessage(content=json.dumps(user_payload, ensure_ascii=False)),
                ]
            )
            queries = self._build_query_list(
                primary_query=parsed.primary_query,
                expansion_queries=parsed.expansion_queries,
                needs_expansion=parsed.needs_expansion,
                user_query=state.user_query,
            )
            if not queries:
                raise ValueError("Planner LLM returned empty queries.")
            reasoning = parsed.reasoning or ["Generated semantic search query variants from user intent."]
            return PlannerOutput(
                planned_query=queries[0],
                queries=queries,
                reasoning=[str(item) for item in reasoning][:5],
                source="llm",
            )
        except Exception as exc:
            LOGGER.warning("Planner LLM failed, using fallback planner: %s", exc)
            state.add_trace(f"Planner LLM failed: {exc}. Using fallback.")
            return self._fallback_planner(state)

    def _fallback_planner(self, state: RagState) -> PlannerOutput:
        optimized_query = self._optimize_query(state)
        return PlannerOutput(
            planned_query=optimized_query,
            queries=[optimized_query],
            reasoning=[
                "Preserved intent-bearing tokens from user input and recent context.",
                "Removed common filler tokens to improve retrieval precision.",
            ],
            source="fallback_python",
        )

    def _optimize_query(self, state: RagState) -> str:
        history_hint = " ".join(msg.content for msg in state.chat_history[-2:] if msg.role == "user")
        source_text = f"{history_hint} {state.user_query}".strip()
        tokens = re.findall(r"[A-Za-z0-9:\-']+", source_text)
        filtered = [t for t in tokens if t.lower() not in _STOPWORDS]
        if filtered:
            return " ".join(filtered)
        return state.user_query.strip()

    def _normalize_queries(self, raw_queries: object) -> list[str]:
        if not isinstance(raw_queries, list):
            return []
        normalized: list[str] = []
        seen: set[str] = set()
        for item in raw_queries:
            text = str(item).strip()
            if not text:
                continue
            key = text.lower()
            if key in seen:
                continue
            seen.add(key)
            normalized.append(text)
            if len(normalized) >= 3:
                break
        return normalized

    def _build_query_list(
        self,
        *,
        primary_query: str,
        expansion_queries: list[str],
        needs_expansion: bool,
        user_query: str,
    ) -> list[str]:
        primary = self._clean_for_retrieval(primary_query.strip() or user_query.strip())
        base_queries = self._normalize_queries([primary])
        if not base_queries:
            return []

        if not needs_expansion:
            heuristic_expansions = self._heuristic_expansions(base_queries[0])
            if heuristic_expansions:
                return self._normalize_queries([*base_queries, *heuristic_expansions])[:3]
            return base_queries

        normalized_expansions = self._normalize_queries([self._clean_for_retrieval(q) for q in expansion_queries])
        combined = self._normalize_queries([*base_queries, *normalized_expansions, *self._heuristic_expansions(base_queries[0])])
        return combined[:3]

    def _clean_for_retrieval(self, query: str) -> str:
        text = query.strip()
        patterns = [
            r"^\s*who\s+is\s+",
            r"^\s*what\s+is\s+",
            r"^\s*tell\s+me\s+about\s+",
            r"^\s*can\s+you\s+explain\s+",
            r"^\s*i\s+want\s+to\s+know\s+about\s+",
        ]
        for pattern in patterns:
            text = re.sub(pattern, "", text, flags=re.IGNORECASE)
        text = re.sub(r"\s+", " ", text).strip(" ?!.")
        return text or query.strip()

    def _heuristic_expansions(self, base_query: str) -> list[str]:
        q = base_query.lower()
        ranking_intent = any(token in q for token in ("strongest", "best", "top", "highest", "most powerful"))
        if not ranking_intent:
            return []

        expansions: list[str] = []
        if "sword" in q:
            expansions.append(re.sub(r"\bsword\b", "weapon", base_query, flags=re.IGNORECASE))
            expansions.append(re.sub(r"\bsword\b", "equipment", base_query, flags=re.IGNORECASE))
        return [e.strip() for e in expansions if e.strip() and e.strip().lower() != base_query.lower()]
