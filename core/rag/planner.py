from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI

from core.env import settings
from core.rag.prompt import MODEL_DICT, PLANNER_PROMPT
from core.rag.schema import PlannerState, RagState


LOGGER = logging.getLogger(__name__)

CONVERSATIONAL_PREFIXES = [
    r"^\s*who\s+is\s+",
    r"^\s*what\s+is\s+",
    r"^\s*what\s+are\s+",
    r"^\s*where\s+(?:is|are|can\s+i\s+find|do\s+i\s+find)\s+",
    r"^\s*when\s+(?:is|was|were|did)\s+",
    r"^\s*tell\s+me\s+about\s+",
    r"^\s*can\s+you\s+(?:explain|tell\s+me\s+about)\s+",
    r"^\s*i\s+want\s+to\s+know\s+about\s+",
]

COMPARISON_MARKERS = {
    "compare",
    "comparison",
    "difference",
    "differences",
    "different",
    "versus",
    " vs ",
    "better than",
}
AMBIGUOUS_REFERENCES = {"he", "she", "they", "it", "this", "that", "them", "those", "these"}
FOLLOW_UP_MARKERS = ("what about", "the other", "that one", "other one", "again")
MULTI_PART_SPLIT = re.compile(r"\s+and\s+", flags=re.IGNORECASE)

ENTITY_PATTERN = re.compile(
    r"\b([A-Z][\w'&-]*(?:\s+(?:[A-Z][\w'&-]*|[IVX]+))*)\b"
)

MAX_ENTITY_COUNT = 12


@dataclass(frozen=True)
class QueryComplexity:
    use_llm: bool
    reason: str


def classify_complexity(query: str, history_count: int) -> QueryComplexity:
    """Decide locally whether a query needs LLM planning.

    Conservative by design: anything unclear is escalated to the LLM.
    """
    text = query.lower()

    for marker in COMPARISON_MARKERS:
        if marker in text:
            return QueryComplexity(use_llm=True, reason=f"comparison_marker:{marker}")

    if query.count("?") > 1:
        return QueryComplexity(use_llm=True, reason="multiple_questions")

    parts = [p for p in MULTI_PART_SPLIT.split(query) if p.strip()]
    if len(parts) > 1 and all(len(p.split()) >= 3 for p in parts):
        return QueryComplexity(use_llm=True, reason="multi_part_request")

    if history_count > 0:
        if any(marker in text for marker in FOLLOW_UP_MARKERS):
            return QueryComplexity(use_llm=True, reason="follow_up_with_history")
        words = set(re.findall(r"[a-z]+", text))
        if words & AMBIGUOUS_REFERENCES:
            return QueryComplexity(use_llm=True, reason="ambiguous_reference_with_history")

    if len(query.split()) > 30:
        return QueryComplexity(use_llm=True, reason="very_long_query")

    return QueryComplexity(use_llm=False, reason="simple_single_intent")


def collapse_to_noun_phrase(query: str) -> str:
    text = query.strip()
    for pattern in CONVERSATIONAL_PREFIXES:
        text = re.sub(pattern, "", text, flags=re.IGNORECASE)
    text = re.sub(r"\s+", " ", text).strip(" ?!.")
    return text or query.strip()


def extract_entities(query: str) -> list[str]:
    """Extract capitalized phrases (names, titles, versions) conservatively."""
    entities: list[str] = []
    seen: set[str] = set()
    for match in ENTITY_PATTERN.finditer(query):
        text = match.group(1).strip()
        if not text or len(text) < 2:
            continue
        key = text.lower()
        if key in seen:
            continue
        seen.add(key)
        entities.append(text)
        if len(entities) >= MAX_ENTITY_COUNT:
            break
    return entities


def dedupe_queries(candidates: list[str]) -> list[str]:
    queries: list[str] = []
    seen: set[str] = set()
    for text in candidates:
        text = text.strip()
        if not text:
            continue
        key = text.lower()
        if key in seen:
            continue
        seen.add(key)
        queries.append(text)
        if len(queries) >= settings.PLANNER_MAX_QUERIES:
            break
    return queries


class ManualPlanner:
    """Deterministic query planning without any LLM call."""

    def plan(self, state: RagState) -> PlannerState:
        raw = state.user_query.strip()
        query = raw
        if not settings.PLANNER_PRESERVE_RICH_QUERY:
            query = collapse_to_noun_phrase(raw)

        planned = dedupe_queries([query]) if query else []

        return PlannerState(
            condensed_query=planned[0] if planned else raw,
            planned_queries=planned,
            entities=extract_entities(raw),
            reasoning=["Deterministic manual planning from the user query."],
            source="manual",
        )


class LlmPlanner:
    """GPT-based query planning with structured output."""

    def __init__(self, recovery: ManualPlanner) -> None:
        self.recovery = recovery
        self.llm = ChatOpenAI(
            model=MODEL_DICT["PLANNER"],
            api_key=settings.OPENAI_API_KEY,
            base_url=settings.OPENAI_BASE_URL or None,
            temperature=0.1,
        )
        self.structured_llm = self.llm.with_structured_output(PlannerState)

    def plan(self, state: RagState) -> PlannerState:
        try:
            return self._invoke_llm(state)
        except Exception as exc:
            LOGGER.warning("Planner LLM failed, using manual planner: %s", exc)
            state.add_trace(f"Planner LLM failed: {exc}. Using manual planner.")
            return self.recovery.plan(state)

    def _invoke_llm(self, state: RagState) -> PlannerState:
        history_hint = [
            {"q": h.q, "a": h.a} for h in state.history[-settings.RAG_HISTORY_WINDOW :]
        ]
        user_payload = {
            "user_query": state.user_query,
            "history": history_hint,
        }
        parsed = self.structured_llm.invoke(
            [
                SystemMessage(content=PLANNER_PROMPT),
                HumanMessage(content=json.dumps(user_payload, ensure_ascii=False)),
            ]
        )

        condensed_query = str(getattr(parsed, "condensed_query", "")).strip()
        if not condensed_query:
            condensed_query = state.user_query.strip()

        raw_queries = [str(q) for q in (parsed.planned_queries or [])]
        if not settings.PLANNER_PRESERVE_RICH_QUERY:
            raw_queries = [collapse_to_noun_phrase(q) for q in raw_queries]
        if not raw_queries and condensed_query:
            raw_queries = [condensed_query]
        planned = dedupe_queries(raw_queries)
        if not planned:
            raise ValueError("Planner LLM returned empty planned_queries.")

        entities: list[str] = []
        for item in list(getattr(parsed, "entities", []) or []):
            text = str(item).strip()
            if text and text.lower() not in {e.lower() for e in entities}:
                entities.append(text)
        entities = entities[:MAX_ENTITY_COUNT]

        reasoning = parsed.reasoning or [
            "Generated semantic search query variants from user intent."
        ]
        return PlannerState(
            condensed_query=condensed_query or planned[0],
            planned_queries=planned,
            entities=entities,
            reasoning=[str(item) for item in reasoning][:5],
            source="llm",
        )


class PlannerNode:
    """Routes each query to the manual planner, the LLM planner, or both.

    Modes (settings.PLANNER_MODE):
      - "manual": always use ManualPlanner
      - "llm":    always use LlmPlanner (manual recovery on failure)
      - "both":   ManualPlanner for simple queries, LlmPlanner otherwise
    """

    def __init__(self) -> None:
        self.manual = ManualPlanner()
        self.llm: LlmPlanner | None = None
        llm_available = bool(settings.PLANNER_AGENT and settings.OPENAI_API_KEY)
        if llm_available and settings.PLANNER_MODE in ("llm", "both"):
            self.llm = LlmPlanner(recovery=self.manual)

    def run(self, state: RagState) -> RagState:
        output = self._plan_with_selected_planner(state)
        state.planner_state = output
        state.add_trace(f"Planner node completed with source={output.source}.")
        return state

    def _plan_with_selected_planner(self, state: RagState) -> PlannerState:
        mode = settings.PLANNER_MODE

        if mode == "manual":
            return self.manual.plan(state)

        if mode == "llm":
            if self.llm is not None:
                return self.llm.plan(state)
            state.add_trace("Planner LLM unavailable. Using manual planner.")
            return self.manual.plan(state)

        # mode == "both"
        complexity = classify_complexity(state.user_query, len(state.history))
        state.add_trace(f"Planner route: mode={mode} reason={complexity.reason}")
        if complexity.use_llm and self.llm is not None:
            return self.llm.plan(state)
        return self.manual.plan(state)
