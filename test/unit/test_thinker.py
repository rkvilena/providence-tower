from __future__ import annotations

from core.rag.prompt import THINKER_PROMPT, THINKER_PROMPT_VERSION
from core.rag.schema import ChunkHit, FetcherState, HistoryTurn, RagState
from core.rag.thinker import ThinkerNode


def make_thinker() -> ThinkerNode:
    thinker = ThinkerNode.__new__(ThinkerNode)
    thinker._cache = None
    return thinker


def make_state(
    query: str,
    *,
    resolved: str = "",
    history: list[HistoryTurn] | None = None,
) -> RagState:
    state = RagState(session_id="test", user_query=query, history=history or [])
    state.planner_state.condensed_query = resolved
    state.fetcher_state = FetcherState(
        chunks=[
            ChunkHit(
                chunk_id="c1",
                page_id="11068",
                page_title="Ryo Takeshita",
                score=0.9,
                text="Ryo Takeshita composed music for Ys Origin.",
            )
        ]
    )
    return state


# ---- Thinker payload ---------------------------------------------------------


def test_thinker_payload_carries_raw_message_resolved_question_and_history():
    history = [
        HistoryTurn(q="Give me an example of Ryo Takeshita music", a="Which game?")
    ]
    state = make_state(
        "Ys Origin",
        resolved="Ryo Takeshita music track title from Ys Origin",
        history=history,
    )
    thinker = make_thinker()

    payload = thinker._build_user_payload(state, chunks_payload=[{"chunk_id": "c1"}])

    assert payload["user_message"] == "Ys Origin"
    assert payload["resolved_question"] == (
        "Ryo Takeshita music track title from Ys Origin"
    )
    assert payload["conversation_history"] == [
        {"q": "Give me an example of Ryo Takeshita music", "a": "Which game?"}
    ]
    assert payload["chunks"] == [{"chunk_id": "c1"}]


def test_thinker_payload_falls_back_to_raw_message_when_unresolved():
    state = make_state("Who is Ryo Takeshita?")
    thinker = make_thinker()

    payload = thinker._build_user_payload(state, chunks_payload=[])

    assert payload["resolved_question"] == "Who is Ryo Takeshita?"


# ---- Thinker cache identity --------------------------------------------------


def test_cache_key_changes_with_resolved_question():
    thinker = make_thinker()
    first = make_state("Ys Origin", resolved="Ys Origin music track titles")
    second = make_state("Ys Origin", resolved="Ys Origin release date")

    assert thinker._cache_key(first) != thinker._cache_key(second)


def test_cache_key_changes_with_conversation_history():
    thinker = make_thinker()
    first = make_state(
        "Ys Origin",
        resolved="Ys Origin music track titles",
        history=[HistoryTurn(q="List his music", a="Which game?")],
    )
    second = make_state(
        "Ys Origin",
        resolved="Ys Origin music track titles",
        history=[HistoryTurn(q="Ys Origin release date", a="2006.")],
    )

    assert thinker._cache_key(first) != thinker._cache_key(second)


# ---- Prompt contract ---------------------------------------------------------


def test_thinker_prompt_requires_answer_first_and_resolved_context():
    assert "resolved_question" in THINKER_PROMPT
    assert "Answer-first policy" in THINKER_PROMPT
    assert "Ask a clarifying question only when" in THINKER_PROMPT
    assert "do not ask when a source-supported interpretation" in THINKER_PROMPT
    assert THINKER_PROMPT_VERSION == "3"
