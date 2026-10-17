"""Unit tests for CrossEncoderReranker batch reranking.

Run from the project root:
    python -m pytest test/unit -q
"""

from __future__ import annotations

import numpy as np

from core.rag.reranker import CrossEncoderReranker
from core.rag.schema import ChunkHit


class FakeModel:
    """Stands in for CrossEncoder; records calls and returns deterministic scores."""

    def __init__(self) -> None:
        self.calls: list[tuple[list[tuple[str, str]], int]] = []

    def predict(self, pairs, batch_size: int = 32, show_progress_bar=None, **kwargs):
        self.calls.append((list(pairs), batch_size))
        # Deterministic: score by passage length so ordering is verifiable.
        return np.array([float(len(text)) for _, text in pairs])


def make_hit(chunk_id: str, text: str, score: float = 0.9) -> ChunkHit:
    return ChunkHit(
        chunk_id=chunk_id,
        page_id="p1",
        page_title="t",
        score=score,
        text=text,
    )


def make_reranker(
    min_top_score: float = 0.7, batch_size: int = 32
) -> CrossEncoderReranker:
    reranker = CrossEncoderReranker.__new__(CrossEncoderReranker)
    reranker.model = FakeModel()
    reranker.min_top_score = min_top_score
    reranker.batch_size = batch_size
    return reranker


def test_rerank_batch_uses_single_model_call():
    reranker = make_reranker()
    q1_hits = [make_hit("a1", "short"), make_hit("a2", "a much longer passage here")]
    q2_hits = [make_hit("b1", "medium length text")]

    results = reranker.rerank_batch([("q1", q1_hits), ("q2", q2_hits)])

    assert len(reranker.model.calls) == 1
    pairs, batch_size = reranker.model.calls[0]
    assert batch_size == 32
    assert pairs == [
        ("q1", "short"),
        ("q1", "a much longer passage here"),
        ("q2", "medium length text"),
    ]
    assert [r.hits[0].chunk_id for r in results] == ["a2", "b1"]


def test_rerank_batch_ranks_and_thresholds_independently():
    reranker = make_reranker(min_top_score=0.7)
    hits = [make_hit("a1", "short"), make_hit("a2", "a much longer passage here")]

    results = reranker.rerank_batch([("q1", hits), ("q2", [])])

    passed, empty = results
    assert [h.chunk_id for h in passed.hits] == ["a2", "a1"]  # sorted desc by score
    assert passed.top_score == len("a much longer passage here")
    assert passed.passed_threshold is True

    assert empty.hits == []
    assert empty.top_score == 0.0
    assert empty.passed_threshold is False


def test_rerank_batch_forwards_configured_batch_size():
    reranker = make_reranker(batch_size=8)
    hits = [make_hit("a1", "x"), make_hit("a2", "yy")]

    reranker.rerank_batch([("q1", hits)])

    _, batch_size = reranker.model.calls[0]
    assert batch_size == 8


def test_rerank_single_query_matches_batch_result():
    reranker = make_reranker()
    hits = [make_hit("a1", "short"), make_hit("a2", "a much longer passage here")]

    single = reranker.rerank("q1", hits)
    batched = reranker.rerank_batch([("q1", hits)])[0]

    assert single.hits == batched.hits
    assert single.top_score == batched.top_score
    assert single.passed_threshold == batched.passed_threshold


def test_rerank_empty_hits():
    reranker = make_reranker()
    result = reranker.rerank("q1", [])
    assert result.hits == []
    assert result.top_score == 0.0
    assert result.passed_threshold is False
    assert reranker.model.calls == []


def test_rerank_batch_failure_threshold_per_query():
    # Below threshold -> passed_threshold=False; the Fetcher keeps the
    # vector-search ordering. The reranker still reports reranked hits.
    reranker = make_reranker(min_top_score=10_000.0)
    hits = [make_hit("a1", "short"), make_hit("a2", "a much longer passage here")]

    result = reranker.rerank("q1", hits)
    assert result.passed_threshold is False
    assert [h.chunk_id for h in result.hits] == ["a2", "a1"]  # reranked order
    assert result.top_score < 10_000.0
