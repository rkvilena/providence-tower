"""Unit tests for Cloudflare-backed RAG reranking.

Run from the project root:
    python -m pytest test/unit -q
"""

from __future__ import annotations

import logging

from core.inference.cloudflare_workers_ai import CloudflareAPIError, RerankScore
from core.rag.fetcher import FetcherNode
from core.rag.reranker import CloudflareReranker
from core.rag.schema import ChunkHit, PlannerState, RagState


class FakeCloudflareClient:
    """Stands in for the shared client and records rerank requests."""

    def __init__(
        self,
        *,
        error: Exception | None = None,
        scores: dict[int, float] | None = None,
    ) -> None:
        self.error = error
        self.scores = scores or {}
        self.calls: list[tuple[str, list[str]]] = []

    def rerank(self, query: str, texts: list[str]) -> list[RerankScore]:
        self.calls.append((query, list(texts)))
        if self.error is not None:
            raise self.error
        return [
            RerankScore(index=index, score=self.scores.get(index, float(len(text))))
            for index, text in enumerate(texts)
        ]


def make_hit(chunk_id: str, text: str, score: float = 0.9) -> ChunkHit:
    return ChunkHit(
        chunk_id=chunk_id,
        page_id="p1",
        page_title="t",
        score=score,
        text=text,
    )


def make_reranker(client: FakeCloudflareClient) -> CloudflareReranker:
    return CloudflareReranker(client=client)


def test_rerank_sends_one_request_per_non_empty_query():
    client = FakeCloudflareClient()
    reranker = make_reranker(client)
    q1_hits = [make_hit("a1", "short"), make_hit("a2", "a much longer passage here")]
    q2_hits = [make_hit("b1", "medium length text")]

    results = reranker.rerank_batch([("q1", q1_hits), ("q2", q2_hits), ("q3", [])])

    assert [call[0] for call in client.calls] == ["q1", "q2"]
    assert client.calls[0][1] == ["short", "a much longer passage here"]
    assert client.calls[1][1] == ["medium length text"]
    assert [result.hits[0].chunk_id for result in results[:2]] == ["a2", "b1"]
    assert results[2].hits == []
    assert results[2].failed is False


def test_rerank_maps_response_indices_back_to_original_hits():
    client = FakeCloudflareClient(scores={0: 0.1, 1: 0.9, 2: 0.5})
    reranker = make_reranker(client)
    hits = [make_hit("a", "alpha"), make_hit("b", "bravo"), make_hit("c", "charlie")]

    result = reranker.rerank("query", hits)

    assert [hit.chunk_id for hit in result.hits] == ["b", "c", "a"]
    assert result.top_score == 0.9
    assert result.failed is False


def test_rerank_keeps_unscored_hits_in_original_order():
    class PartialClient(FakeCloudflareClient):
        def rerank(self, query: str, texts: list[str]) -> list[RerankScore]:
            self.calls.append((query, list(texts)))
            return [RerankScore(index=1, score=0.9)]

    reranker = make_reranker(PartialClient())
    hits = [make_hit("a", "alpha"), make_hit("b", "bravo"), make_hit("c", "charlie")]

    result = reranker.rerank("query", hits)

    assert [hit.chunk_id for hit in result.hits] == ["b", "a", "c"]


def test_rerank_failure_retains_vector_search_order(caplog):
    client = FakeCloudflareClient(error=CloudflareAPIError("upstream unavailable"))
    reranker = make_reranker(client)
    hits = [make_hit("a1", "short"), make_hit("a2", "longer")]

    with caplog.at_level(logging.WARNING):
        result = reranker.rerank("query", hits)

    assert result.hits == hits
    assert result.failed is True
    assert "upstream unavailable" in result.error
    assert "Cloudflare rerank failed" in caplog.text


def test_rerank_empty_hits_skips_the_cloud_call():
    client = FakeCloudflareClient()
    reranker = make_reranker(client)

    result = reranker.rerank("query", [])

    assert result.hits == []
    assert result.top_score == 0.0
    assert client.calls == []


def test_rerank_single_query_matches_batch_result():
    client = FakeCloudflareClient(scores={0: 0.2, 1: 0.8})
    reranker = make_reranker(client)
    hits = [make_hit("a1", "short"), make_hit("a2", "longer")]

    single = reranker.rerank("query", hits)
    batched = reranker.rerank_batch([("query", hits)])[0]

    assert single == batched


def make_fetcher(reranker) -> FetcherNode:
    fetcher = FetcherNode.__new__(FetcherNode)
    fetcher.store = None
    fetcher.cloudflare = None
    fetcher.reranker = reranker
    return fetcher


def test_fetcher_reports_one_model_call_per_non_empty_query():
    client = FakeCloudflareClient()
    fetcher = make_fetcher(make_reranker(client))
    state = RagState(
        session_id="s",
        user_query="q",
        planner_state=PlannerState(planned_queries=["q1", "q2"]),
    )
    hits_by_query = [
        ("q1", [make_hit("a1", "short")]),
        ("q2", [make_hit("b1", "other")]),
        ("q3", []),
    ]

    outcome = fetcher._rerank_and_merge(state, hits_by_query, {}, {})

    assert outcome.model_calls == 2
    assert len(client.calls) == 2
    assert outcome.pair_count == 2


def test_fetcher_keeps_original_order_when_rerank_fails():
    client = FakeCloudflareClient(error=CloudflareAPIError("bad gateway"))
    fetcher = make_fetcher(make_reranker(client))
    state = RagState(
        session_id="s",
        user_query="q",
        planner_state=PlannerState(planned_queries=["q1"]),
    )
    original_hits = [make_hit("a1", "short"), make_hit("a2", "longer", score=0.1)]

    outcome = fetcher._rerank_and_merge(state, [("q1", original_hits)], {}, {})

    assert outcome.model_calls == 1
    assert any(
        "Rerank failed for query 'q1'" in trace and "vector-search order" in trace
        for trace in state.debug_trace
    )
