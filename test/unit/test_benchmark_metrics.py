"""Unit tests for benchmark retrieval and ranking metrics."""

from __future__ import annotations

from pathlib import Path

from test.benchmark_runner import compute_retrieval_metrics
from test.benchmark_schemas import (
    BenchmarkResult,
    RetrievalMetrics,
    summarize,
)


def test_mrr_uses_the_first_relevant_rank():
    metrics = compute_retrieval_metrics(["a"], ["x", "a", "y"])

    assert metrics.precision == round(1 / 3, 4)
    assert metrics.recall == 1.0
    assert metrics.hit is True
    assert metrics.full_recall is True
    assert metrics.mrr == 0.5
    assert metrics.recall_at_5 == 1.0


def test_recall_at_5_ignores_relevant_results_beyond_the_cutoff():
    metrics = compute_retrieval_metrics(
        ["a"], ["x1", "x2", "x3", "x4", "x5", "a"]
    )

    assert metrics.hit is True
    assert metrics.full_recall is True
    assert metrics.mrr == round(1 / 6, 4)
    assert metrics.recall_at_5 == 0.0


def test_ranking_metrics_use_the_first_occurrence_of_duplicate_ids():
    metrics = compute_retrieval_metrics(["a", "b"], ["a", "a", "b"])

    assert metrics.mrr == 1.0
    assert metrics.recall_at_5 == 1.0
    assert metrics.precision == 1.0


def test_empty_labels_produce_zero_ranking_metrics():
    metrics = compute_retrieval_metrics([], ["a"])

    assert metrics.precision == 0.0
    assert metrics.recall == 0.0
    assert metrics.hit is False
    assert metrics.full_recall is False
    assert metrics.mrr == 0.0
    assert metrics.recall_at_5 == 0.0


def test_no_relevant_result_produces_zero_ranking_metrics():
    metrics = compute_retrieval_metrics(["a"], ["x", "y"])

    assert metrics.hit is False
    assert metrics.full_recall is False
    assert metrics.mrr == 0.0
    assert metrics.recall_at_5 == 0.0


def test_summary_averages_ranking_metrics_over_successful_results():
    results = [
        BenchmarkResult(
            id=1,
            question="q1",
            expected_context_chunk_ids=["a"],
            ground_truth="",
            retrieval=RetrievalMetrics(mrr=1.0, recall_at_5=1.0),
        ),
        BenchmarkResult(
            id=2,
            question="q2",
            expected_context_chunk_ids=["b"],
            ground_truth="",
            retrieval=RetrievalMetrics(mrr=0.5, recall_at_5=0.0),
        ),
        BenchmarkResult(
            id=3,
            question="q3",
            expected_context_chunk_ids=["c"],
            ground_truth="",
            status="error",
            retrieval=RetrievalMetrics(mrr=0.0, recall_at_5=0.0),
        ),
    ]

    summary = summarize(
        run_id="test",
        dataset_path=Path("data.jsonl"),
        dataset_hash="hash",
        results=results,
    )

    assert summary.retrieval.mean_mrr == 0.75
    assert summary.retrieval.mean_recall_at_5 == 0.5
