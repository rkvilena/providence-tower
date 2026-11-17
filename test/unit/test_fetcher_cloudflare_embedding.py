"""Unit tests for Fetcher query embedding via Cloudflare Workers AI.

Run from the project root:
    python -m pytest test/unit -q
"""

from __future__ import annotations

import numpy as np
import pytest

from core.inference.cloudflare_workers_ai import (
    EMBEDDING_DIMENSIONS,
    CloudflareAPIError,
    CloudflareConfigurationError,
)
from core.rag.fetcher import FetcherNode
from core.rag.schema import PlannerState, QueryFetchTiming, RagState


def cloudflare_vector(value: float = 0.25) -> list[float]:
    return [value] * EMBEDDING_DIMENSIONS


class FakeCloudflareClient:
    def __init__(
        self,
        vectors: dict[str, list[float]] | None = None,
        error: Exception | None = None,
    ) -> None:
        self.vectors = vectors or {}
        self.error = error
        self.queries: list[str] = []

    def embed_query(self, text: str) -> list[float]:
        self.queries.append(text)
        if self.error is not None:
            raise self.error
        return self.vectors.get(text, cloudflare_vector())


class FakeStore:
    def __init__(self) -> None:
        self.search_calls: list[np.ndarray] = []

    def search_hybrid(self, query_vector, *, top_k=20, filter_entities=None):
        self.search_calls.append(query_vector)
        return []


def make_fetcher(
    *, client: FakeCloudflareClient | None = None, store: FakeStore | None = None
) -> FetcherNode:
    fetcher = FetcherNode.__new__(FetcherNode)
    fetcher.store = store or FakeStore()
    fetcher.cloudflare = client or FakeCloudflareClient()
    fetcher.reranker = None
    return fetcher


def make_state(*queries: str) -> RagState:
    return RagState(
        session_id="session-1",
        user_query="question",
        planner_state=PlannerState(planned_queries=list(queries)),
    )


def test_search_query_embeds_with_cloudflare_and_returns_float32():
    client = FakeCloudflareClient({"Who is Marco?": cloudflare_vector(0.5)})
    store = FakeStore()
    fetcher = make_fetcher(client=client, store=store)

    hits = fetcher._search_query(
        "Who is Marco?", [], QueryFetchTiming(query="Who is Marco?")
    )

    assert hits == []
    assert client.queries == ["Who is Marco?"]
    [sent_vector] = store.search_calls
    assert sent_vector.dtype == np.float32
    assert sent_vector.shape == (EMBEDDING_DIMENSIONS,)
    assert np.allclose(sent_vector, 0.5)


def test_search_query_rejects_wrong_dimensions_without_searching():
    client = FakeCloudflareClient({"query": cloudflare_vector()[:-1]})
    store = FakeStore()
    fetcher = make_fetcher(client=client, store=store)

    with pytest.raises(CloudflareAPIError, match="384"):
        fetcher._search_query("query", [], QueryFetchTiming(query="query"))
    assert store.search_calls == []


@pytest.mark.parametrize(
    "bad_vector",
    [
        [[0.1] * EMBEDDING_DIMENSIONS],  # 2-D instead of a single vector
        [float("nan")] * EMBEDDING_DIMENSIONS,
    ],
)
def test_search_query_rejects_malformed_vectors(bad_vector):
    fetcher = make_fetcher(client=FakeCloudflareClient({"query": bad_vector}))

    with pytest.raises(CloudflareAPIError):
        fetcher._search_query("query", [], QueryFetchTiming(query="query"))


def test_query_failure_is_traced_and_skipped_without_local_fallback():
    client = FakeCloudflareClient(
        error=CloudflareConfigurationError("Cloudflare is not configured")
    )
    fetcher = make_fetcher(client=client)
    state = make_state("first query", "second query")

    timings, hits_by_query = fetcher._search_all_queries(
        state, ["first query", "second query"], []
    )

    assert client.queries == ["first query", "second query"]
    assert len(timings) == 2
    assert hits_by_query == []
    assert any("Cloudflare is not configured" in trace for trace in state.debug_trace)
    assert len(state.debug_trace) >= 2


def test_search_all_queries_skips_only_failing_query():
    class FlakyClient(FakeCloudflareClient):
        def embed_query(self, text: str) -> list[float]:
            if text == "bad":
                raise CloudflareAPIError("upstream failure")
            return super().embed_query(text)

    fetcher = make_fetcher(client=FlakyClient())
    state = make_state("bad", "good")

    timings, hits_by_query = fetcher._search_all_queries(state, ["bad", "good"], [])

    assert [timing.query for timing in timings] == ["bad", "good"]
    assert len(hits_by_query) == 1
    assert hits_by_query[0][0] == "good"
    assert any("upstream failure" in trace for trace in state.debug_trace)
