"""Tests for the per-run Markdown benchmark report."""

from __future__ import annotations

import json

import pytest

from test.benchmark_report import generate_report


def write_json(path, data) -> None:
    path.write_text(json.dumps(data, indent=2), encoding="utf-8")


def write_jsonl(path, rows) -> None:
    path.write_text(
        "\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8"
    )


def make_summary(run_id: str = "test-run") -> dict:
    return {
        "run_id": run_id,
        "dataset_path": "test/GOLDEN-DATASET.jsonl",
        "dataset_sha256": "a" * 64,
        "query_count": 2,
        "success_count": 1,
        "failure_count": 1,
        "planner_model": "test-model",
        "reranker_enabled": True,
        "started_at": "2026-01-01T00:00:00+00:00",
        "completed_at": "2026-01-01T00:00:01+00:00",
        "selection": {"mode": "all", "ids": [1, 2]},
        "run_metadata": {
            "code_revision": "abc123def456",
            "code_dirty": False,
            "corpus_version": None,
            "index_version": None,
            "embedding_model": "embed-model",
            "query_embedding_model": "query-embed-model",
            "reranker_model": "rerank-model",
            "generation_model": "generation-model",
        },
        "performance": {
            "mean_ms": 100.0,
            "median_ms": 100.0,
            "p95_ms": 100.0,
            "min_ms": 100.0,
            "max_ms": 100.0,
        },
        "retrieval": {
            "mean_precision": 0.5,
            "mean_recall": 1.0,
            "hit_rate": 1.0,
            "full_recall_rate": 1.0,
            "mean_mrr": 1.0,
            "mean_recall_at_5": 1.0,
        },
        "tokens": {"available": False, "mean_total_tokens": None, "total_tokens": 0},
    }


def make_raw_rows() -> list[dict]:
    return [
        {
            "id": 1,
            "question": "Which weapon does the character use?",
            "status": "success",
            "expected_context_chunk_ids": ["a"],
            "retrieved_chunk_ids": ["a", "b"],
            "candidate_chunk_ids_by_query": {"q1": ["a", "b"]},
            "actual_answer": "The character uses claws.",
            "total_execution_ms": 100.0,
            "retrieval": {
                "precision": 0.5,
                "recall": 1.0,
                "hit": True,
                "full_recall": True,
                "mrr": 1.0,
                "recall_at_5": 1.0,
            },
        },
        {
            "id": 2,
            "question": "Which item is missing?",
            "status": "error",
            "expected_context_chunk_ids": ["c"],
            "retrieved_chunk_ids": [],
            "candidate_chunk_ids_by_query": {},
            "actual_answer": "",
            "total_execution_ms": 50.0,
            "retrieval": {
                "precision": 0.0,
                "recall": 0.0,
                "hit": False,
                "full_recall": False,
                "mrr": 0.0,
                "recall_at_5": 0.0,
            },
            "error": {
                "type": "RuntimeError",
                "message": "service unavailable",
                "traceback": "trace",
            },
        },
    ]


def make_quality_row() -> dict:
    return {
        "id": 1,
        "status": "success",
        "checks_total": 1,
        "checks_passed": 1,
        "answer_checks_status": "pass",
        "answer_checks_reason": None,
        "checks": [{"label": "names claws", "passed": True}],
        "token_overlap": 0.75,
    }


def make_quality_summary() -> dict:
    return {
        "questions": 2,
        "evaluated": 1,
        "unevaluated": 1,
        "total_checks": 1,
        "passed_checks": 1,
        "check_pass_rate": 1.0,
        "mean_token_overlap": 0.75,
        "note": "Checks cover only reviewed facts; token overlap is diagnostic.",
    }


def test_report_includes_tables_questions_checks_and_failures(tmp_path):
    write_json(tmp_path / "summary.json", make_summary())
    write_jsonl(tmp_path / "raw-results.jsonl", make_raw_rows())
    write_jsonl(tmp_path / "quality-results.jsonl", [make_quality_row()])
    write_json(tmp_path / "quality-summary.json", make_quality_summary())

    report_path = generate_report(tmp_path)
    report = report_path.read_text(encoding="utf-8")

    assert report_path == tmp_path / "report.md"
    assert "# RAG Benchmark Report: test-run" in report
    assert "| MRR | Recall@5 |" in report
    assert "### [1] Which weapon does the character use?" in report
    assert "**Pre-rerank candidate pool:** `q1` -> a, b" in report
    assert "- [x] names claws" in report
    assert "**Received" not in report
    assert "**[[1]]" not in report
    assert "**[2] Run failed:** RuntimeError: service unavailable" in report
    assert "| Check pass rate | 100.0% |" in report
    assert "| Code revision | `abc123def456` |" in report
    assert "| Tracked working-tree changes | no |" in report
    assert "| Corpus version | unavailable |" in report
    assert "| Generation model | `generation-model` |" in report


def test_report_explains_when_answer_checks_have_not_run(tmp_path):
    write_json(tmp_path / "summary.json", make_summary("no-quality"))
    write_jsonl(tmp_path / "raw-results.jsonl", make_raw_rows())

    report = generate_report(tmp_path).read_text(encoding="utf-8")

    assert "Answer checks have not been run for this run." in report
    assert "not run" in report


def test_report_requires_runner_outputs(tmp_path):
    with pytest.raises(FileNotFoundError):
        generate_report(tmp_path)
