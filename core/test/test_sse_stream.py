"""Focused tests for the phase-progress SSE endpoint (no pytest dependency).

Run with::

    python -m core.test.test_sse_stream

Covers:

1. ``RagStreamEvent`` SSE wire-format encoding.
2. ``build_chat_response`` mapping used by both ``/rag/chat`` and the
   ``final_answer`` SSE event.
3. ``RagService.stream_rag`` event ordering and payloads (with a stub graph).
4. ``pipeline_error`` emission on pipeline failure.
5. The ``/rag/chat/stream`` route through ``TestClient`` with dependency
   overrides (no Redis / model access required).
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

# Allow direct script execution: `python core/test/test_sse_stream.py`
if __package__ is None or __package__ == "":
    project_root = Path(__file__).resolve().parents[2]
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))

from core.api.schemas import RagStreamEvent, build_chat_response
from core.api.service import RagService
from core.rag.schema import (
    FetcherState,
    PlannerState,
    RagState,
    ThinkerState,
)

EXPECTED_EVENT_ORDER = [
    "pipeline_started",
    "planner_started",
    "planner_completed",
    "retrieval_started",
    "retrieval_completed",
    "generation_started",
    "generation_completed",
    "context_started",
    "context_completed",
    "final_answer",
    "pipeline_completed",
]

PHASE_BY_NODE = {"planner": "planner", "fetcher": "retrieval", "thinker": "generation", "context": "context"}


# --------------------------------------------------------------------------- #
# Helpers                                                                     #
# --------------------------------------------------------------------------- #


def _parse_sse_stream(raw: str) -> list[tuple[str, dict]]:
    """Parse an SSE body into ``[(event_name, data_dict), ...]``."""
    events: list[tuple[str, dict]] = []
    for frame in raw.split("\n\n"):
        if not frame.strip():
            continue
        name = None
        data: dict = {}
        for line in frame.split("\n"):
            if line.startswith("event: "):
                name = line[len("event: "):]
            elif line.startswith("data: "):
                data = json.loads(line[len("data: "):])
        assert name is not None, f"SSE frame missing event name: {frame!r}"
        events.append((name, data))
    return events


def _advance(state: RagState, node: str) -> RagState:
    """Return a copy of ``state`` as it would look after ``node`` completed."""
    if node == "planner":
        state.planner_state = PlannerState(
            condensed_query="adol abilities",
            planned_queries=["adol abilities"],
            source="fallback_python",
        )
    elif node == "fetcher":
        state.fetcher_state = FetcherState()  # empty chunks keeps payload small
    elif node == "thinker":
        state.thinker_state = ThinkerState(
            sufficient=True, response="Adol is the protagonist."
        )
    state.node_latencies_ms[node] = 1.23
    return state


class _StubRag:
    """Stands in for ``RagGraph``: replays the four-node lifecycle."""

    def __init__(self, *, fail_on: str | None = None) -> None:
        self.fail_on = fail_on

    def stream(self, state: RagState):
        for node in ("planner", "fetcher", "thinker", "context"):
            if node == self.fail_on:
                raise RuntimeError("boom")
            yield node, _advance(state, node)


def _make_service(fail_on: str | None = None) -> RagService:
    svc = RagService()
    svc._history_store = None
    svc._rag = _StubRag(fail_on=fail_on)  # type: ignore[assignment]
    return svc


# --------------------------------------------------------------------------- #
# Tests                                                                       #
# --------------------------------------------------------------------------- #


def test_sse_encode_wire_format() -> None:
    event = RagStreamEvent(event="planner_started", data={"session_id": "s1"})
    encoded = event.encode()
    assert encoded == 'event: planner_started\ndata: {"session_id": "s1"}\n\n'


def test_build_chat_response_mapping() -> None:
    state = RagState(session_id="s1", user_query="q")
    _advance(state, "planner")
    _advance(state, "fetcher")
    _advance(state, "thinker")
    resp = build_chat_response(state)
    assert resp.session_id == "s1"
    assert resp.response == "Adol is the protagonist."
    assert resp.sufficient is True
    assert resp.planner.planned_queries == ["adol abilities"]
    assert resp.node_latencies_ms["thinker"] == 1.23


def test_stream_rag_event_order_and_payloads() -> None:
    svc = _make_service()
    events = _parse_sse_stream(
        "".join(e.encode() for e in svc.stream_rag(query="q", session_id="s1"))
    )

    assert [name for name, _ in events] == EXPECTED_EVENT_ORDER

    planner_completed = dict(events)["planner_completed"]
    assert planner_completed["planned_query_count"] == 1
    assert planner_completed["source"] == "fallback_python"
    assert planner_completed["latency_ms"] == 1.23

    final = dict(events)["final_answer"]
    # The final_answer payload must equal the /rag/chat response schema.
    assert final["session_id"] == "s1"
    assert final["response"] == "Adol is the protagonist."
    assert final["chunk_count"] == 0
    assert final["planner"]["source"] == "fallback_python"


def test_stream_rag_error_event() -> None:
    svc = _make_service(fail_on="fetcher")
    events = _parse_sse_stream(
        "".join(e.encode() for e in svc.stream_rag(query="q", session_id="s1"))
    )
    names = [name for name, _ in events]
    # Fails inside fetcher: planner completed, retrieval started, then error.
    assert names == [
        "pipeline_started",
        "planner_started",
        "planner_completed",
        "retrieval_started",
        "pipeline_error",
    ]
    err = dict(events)["pipeline_error"]
    assert err["error_type"] == "RuntimeError"
    assert "boom" in err["detail"]


def test_chat_stream_route() -> None:
    from fastapi.testclient import TestClient

    from core.api.dependencies import get_rag_service
    from core.api.main import create_app
    from core.api.rate_limiter import get_redis_rate_limiter

    app = create_app()
    app.dependency_overrides[get_rag_service] = lambda: _make_service()
    app.dependency_overrides[get_redis_rate_limiter] = lambda: None

    client = TestClient(app)
    resp = client.post(
        "/rag/chat/stream",
        json={"query": "q", "session_id": "s1"},
        headers={"Accept": "text/event-stream"},
    )
    assert resp.status_code == 200
    assert resp.headers["content-type"].startswith("text/event-stream")
    assert "no-cache" in resp.headers["cache-control"]

    events = _parse_sse_stream(resp.text)
    assert [name for name, _ in events] == EXPECTED_EVENT_ORDER
    final = dict(events)["final_answer"]
    assert final["response"] == "Adol is the protagonist."


def main() -> None:
    tests = [
        test_sse_encode_wire_format,
        test_build_chat_response_mapping,
        test_stream_rag_event_order_and_payloads,
        test_stream_rag_error_event,
        test_chat_stream_route,
    ]
    failed = 0
    for test in tests:
        try:
            test()
            print(f"PASS {test.__name__}")
        except Exception as exc:  # noqa: BLE001 - report and continue
            failed += 1
            print(f"FAIL {test.__name__}: {type(exc).__name__}: {exc}")
    if failed:
        raise SystemExit(1)
    print(f"\nAll {len(tests)} tests passed.")


if __name__ == "__main__":
    main()