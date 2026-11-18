from __future__ import annotations

from core.env import settings
from core.rag.planner import (
    LlmPlanner,
    ManualPlanner,
    PlannerNode,
    collapse_to_noun_phrase,
    classify_complexity,
    dedupe_queries,
    extract_entities,
)
from core.rag.schema import HistoryTurn, PlannerState, RagState


import pytest


@pytest.fixture(autouse=True)
def restore_planner_settings():
    snapshot = (settings.PLANNER_MODE, settings.PLANNER_AGENT)
    yield
    settings.PLANNER_MODE, settings.PLANNER_AGENT = snapshot


def make_state(query: str, history: list[HistoryTurn] | None = None) -> RagState:
    return RagState(session_id="test", user_query=query, history=history or [])


# ---- shared helpers ---------------------------------------------------------


def test_collapse_removes_conversational_prefix():
    assert collapse_to_noun_phrase("Who is Geis in Ys?") == "Geis in Ys"


def test_collapse_preserves_relationship_words():
    result = collapse_to_noun_phrase("What weapon does Toal use?")
    assert "weapon" in result.lower()
    assert "Toal" in result


def test_dedupe_queries_is_case_insensitive_and_capped():
    original_max = settings.PLANNER_MAX_QUERIES
    settings.PLANNER_MAX_QUERIES = 2
    try:
        result = dedupe_queries(["Geis Ys", "geis ys", "Dogi Ys", "Toal Ys"])
    finally:
        settings.PLANNER_MAX_QUERIES = original_max
    assert result == ["Geis Ys", "Dogi Ys"]


def test_extract_entities_finds_names_and_versions():
    entities = extract_entities("Which games feature Geis in Ys Seven?")
    lowered = [e.lower() for e in entities]
    assert any("geis" in e for e in lowered)
    assert any("ys" in e for e in lowered)


# ---- complexity classification ----------------------------------------------


def test_simple_query_routes_to_manual():
    result = classify_complexity("Who is Geis in Ys?", history_count=0)
    assert result.use_llm is False
    assert result.reason == "simple_single_intent"


def test_comparison_query_routes_to_llm():
    result = classify_complexity(
        "How are Geis and Dogi different in Ys Seven?", history_count=0
    )
    assert result.use_llm is True
    assert "comparison" in result.reason


def test_multi_part_query_routes_to_llm():
    result = classify_complexity(
        "What weapon does Toal use and when did he join the Darklings?",
        history_count=0,
    )
    assert result.use_llm is True
    assert result.reason == "multi_part_request"


def test_follow_up_with_history_routes_to_llm():
    result = classify_complexity("What about the other one?", history_count=2)
    assert result.use_llm is True
    assert result.reason == "follow_up_with_history"


def test_ambiguous_reference_with_history_routes_to_llm():
    result = classify_complexity("Where did he find it?", history_count=1)
    assert result.use_llm is True
    assert result.reason == "ambiguous_reference_with_history"


def test_ambiguous_reference_without_history_stays_manual():
    result = classify_complexity("Where can I find it?", history_count=0)
    assert result.use_llm is False


def test_possessive_reference_with_history_routes_to_llm():
    result = classify_complexity("List me some of his music", history_count=1)
    assert result.use_llm is True
    assert result.reason == "ambiguous_reference_with_history"


def test_short_clarification_reply_with_history_routes_to_llm():
    result = classify_complexity("Ys Origin", history_count=1)
    assert result.use_llm is True
    assert result.reason == "clarification_reply_with_history"


def test_short_reply_without_history_stays_manual():
    result = classify_complexity("Ys Origin", history_count=0)
    assert result.use_llm is False


def test_short_question_with_history_stays_manual():
    result = classify_complexity("What is Ys?", history_count=1)
    assert result.use_llm is False


def test_classify_complexity_accepts_history_turns():
    history = [HistoryTurn(q="Who is Ryo Takeshita?", a="A composer.")]
    result = classify_complexity("Ys Origin", history)
    assert result.use_llm is True
    assert result.reason == "clarification_reply_with_history"


def test_planner_llm_payload_includes_history_and_context_rules():
    captured: dict = {}

    class FakeStructuredLlm:
        def invoke(self, messages):
            captured["messages"] = messages
            return PlannerState(
                condensed_query="Ys Origin music track titles",
                planned_queries=["Ys Origin music track titles"],
                source="llm",
            )

    planner = LlmPlanner.__new__(LlmPlanner)
    planner.structured_llm = FakeStructuredLlm()
    state = make_state(
        "Ys Origin",
        history=[HistoryTurn(q="Give me a music title", a="Which game?")],
    )

    result = planner._invoke_llm(state)

    assert result.condensed_query == "Ys Origin music track titles"
    system_message, human_message = captured["messages"]
    assert "possessives" in system_message.content
    assert "short reply" in system_message.content
    assert '"history": [{"q": "Give me a music title", "a": "Which game?"}]' in (
        human_message.content
    )


# ---- ManualPlanner -----------------------------------------------------------


def test_manual_planner_produces_clean_query():
    planner = ManualPlanner()
    state = make_state("Who is Geis in Ys?")
    plan = planner.plan(state)
    assert plan.source == "manual"
    assert plan.planned_queries == ["Geis in Ys"]
    assert plan.condensed_query == "Geis in Ys"
    assert plan.entities


def test_manual_planner_empty_query():
    planner = ManualPlanner()
    state = make_state("   ")
    plan = planner.plan(state)
    assert plan.planned_queries == []
    assert plan.source == "manual"


def test_manual_planner_preserves_rich_query_when_configured():
    original = settings.PLANNER_PRESERVE_RICH_QUERY
    settings.PLANNER_PRESERVE_RICH_QUERY = True
    try:
        planner = ManualPlanner()
        state = make_state("Who is Geis in Ys?")
        plan = planner.plan(state)
    finally:
        settings.PLANNER_PRESERVE_RICH_QUERY = original
    assert plan.planned_queries == ["Who is Geis in Ys?"]


# ---- PlannerNode routing -----------------------------------------------------


class FakeLlmPlanner:
    def __init__(self):
        self.calls: list[str] = []

    def plan(self, state: RagState) -> PlannerState:
        self.calls.append(state.user_query)
        return PlannerState(
            condensed_query=state.user_query,
            planned_queries=[state.user_query],
            source="llm",
        )


def build_node(mode: str) -> PlannerNode:
    settings.PLANNER_MODE = mode
    original_agent = settings.PLANNER_AGENT
    settings.PLANNER_AGENT = False  # skip real LLM construction
    try:
        node = PlannerNode()
    finally:
        settings.PLANNER_AGENT = original_agent
    return node


def test_mode_manual_always_uses_manual_planner():
    node = build_node("manual")
    node.llm = FakeLlmPlanner()

    simple = node.run(make_state("Who is Geis in Ys?"))
    complex_state = make_state("How are Geis and Dogi different in Ys Seven?")
    complex_result = node.run(complex_state)

    assert simple.planner_state.source == "manual"
    assert complex_result.planner_state.source == "manual"
    assert node.llm.calls == []


def test_mode_llm_uses_llm_planner():
    node = build_node("llm")
    fake = FakeLlmPlanner()
    node.llm = fake

    state = make_state("Who is Geis in Ys?")
    result = node.run(state)

    assert result.planner_state.source == "llm"
    assert fake.calls == ["Who is Geis in Ys?"]


def test_mode_llm_without_llm_falls_back_to_manual():
    node = build_node("llm")
    node.llm = None

    result = node.run(make_state("Who is Geis in Ys?"))
    assert result.planner_state.source == "manual"


def test_mode_both_routes_simple_to_manual():
    node = build_node("both")
    fake = FakeLlmPlanner()
    node.llm = fake

    result = node.run(make_state("Who is Geis in Ys?"))
    assert result.planner_state.source == "manual"
    assert fake.calls == []


def test_mode_both_routes_complex_to_llm():
    node = build_node("both")
    fake = FakeLlmPlanner()
    node.llm = fake

    state = make_state("How are Geis and Dogi different in Ys Seven?")
    result = node.run(state)
    assert result.planner_state.source == "llm"
    assert fake.calls == [state.user_query]


def test_mode_both_routes_follow_up_with_history_to_llm():
    node = build_node("both")
    fake = FakeLlmPlanner()
    node.llm = fake

    state = make_state(
        "Ys Origin",
        history=[HistoryTurn(q="Give me a music title", a="Which game?")],
    )
    result = node.run(state)

    assert result.planner_state.source == "llm"
    assert fake.calls == ["Ys Origin"]


def test_mode_both_without_llm_uses_manual_for_complex():
    node = build_node("both")
    node.llm = None

    state = make_state("How are Geis and Dogi different in Ys Seven?")
    result = node.run(state)
    assert result.planner_state.source == "manual"
