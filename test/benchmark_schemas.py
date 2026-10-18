"""Benchmark schemas, dataset helpers, and aggregation for the RAG test suite."""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Literal

TEST_DIR = Path(__file__).resolve().parent


def now_utc_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


@dataclass
class TokenUsage:
    available: bool = False
    prompt_tokens: int | None = None
    completion_tokens: int | None = None
    total_tokens: int | None = None


@dataclass
class RetrievalMetrics:
    precision: float = 0.0
    recall: float = 0.0
    hit: bool = False
    full_recall: bool = False


@dataclass
class QualityMetrics:
    answer_relevancy: float | None = None
    faithfulness: float | None = None
    semantic_similarity: float | None = None


@dataclass
class BenchmarkResult:
    id: int
    question: str
    expected_context_chunk_ids: list[str]
    ground_truth: str
    status: Literal["success", "error"] = "success"
    error: dict[str, Any] | None = None
    actual_answer: str = ""
    total_execution_ms: float = 0.0
    retrieved_chunk_ids: list[str] = field(default_factory=list)
    used_chunk_ids: list[str] = field(default_factory=list)
    used_chunk_recall: float = 0.0
    retrieval: RetrievalMetrics = field(default_factory=RetrievalMetrics)
    quality: QualityMetrics = field(default_factory=QualityMetrics)
    node_latencies_ms: dict[str, float] = field(default_factory=dict)
    fetcher_timing: dict[str, Any] | None = None
    thinker_timing: dict[str, Any] | None = None
    token_usage: TokenUsage = field(default_factory=TokenUsage)
    created_at: str = field(default_factory=now_utc_iso)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class PerformanceSummary:
    mean_ms: float = 0.0
    median_ms: float = 0.0
    p95_ms: float = 0.0
    min_ms: float = 0.0
    max_ms: float = 0.0


@dataclass
class RetrievalSummary:
    mean_precision: float = 0.0
    mean_recall: float = 0.0
    hit_rate: float = 0.0
    full_recall_rate: float = 0.0


@dataclass
class QualitySummary:
    mean_answer_relevancy: float | None = None
    mean_faithfulness: float | None = None
    mean_semantic_similarity: float | None = None


@dataclass
class TokensSummary:
    available: bool = False
    mean_total_tokens: float | None = None
    total_tokens: int = 0


@dataclass
class BenchmarkSummary:
    run_id: str
    dataset_path: str
    dataset_sha256: str
    query_count: int
    success_count: int
    failure_count: int
    model: str = ""
    planner_model: str = ""
    reranker_enabled: bool = False
    execution_mode: str = "sequential"
    started_at: str = ""
    completed_at: str = ""
    dataset_size: int = 0
    selection: dict[str, Any] | None = None
    performance: PerformanceSummary = field(default_factory=PerformanceSummary)
    retrieval: RetrievalSummary = field(default_factory=RetrievalSummary)
    quality: QualitySummary = field(default_factory=QualitySummary)
    tokens: TokensSummary = field(default_factory=TokensSummary)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def save_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def save_jsonl(path: Path, items: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for item in items:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")


def file_sha256(path: Path) -> str:
    sha = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            sha.update(chunk)
    return sha.hexdigest()


def percentile(values: list[float], pct: float) -> float:
    """Linear-interpolated percentile; empty list returns 0.0."""
    if not values:
        return 0.0
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    k = (len(ordered) - 1) * (pct / 100.0)
    lo = int(k)
    hi = min(lo + 1, len(ordered) - 1)
    frac = k - lo
    return ordered[lo] * (1 - frac) + ordered[hi] * frac


def parse_expected_chunk_ids(raw: str) -> list[str]:
    """Normalize 'a, b' style context_chunk field into a list of ids."""
    return [part.strip() for part in raw.split(",") if part.strip()]


def summarize(
    run_id: str,
    dataset_path: Path,
    dataset_hash: str,
    results: list[BenchmarkResult],
    *,
    model: str = "",
    planner_model: str = "",
    reranker_enabled: bool = False,
    started_at: str = "",
    completed_at: str = "",
    selection: dict[str, Any] | None = None,
    dataset_size: int = 0,
) -> BenchmarkSummary:
    """Aggregate per-query results into a run-level summary."""
    ok = [r for r in results if r.status == "success"]
    times = [r.total_execution_ms for r in ok]

    def mean(vals: list[float]) -> float:
        return sum(vals) / len(vals) if vals else 0.0

    def mean_opt(vals: list[float | None]) -> float | None:
        present = [v for v in vals if v is not None]
        return sum(present) / len(present) if present else None

    token_totals = [
        r.token_usage.total_tokens for r in ok if r.token_usage.total_tokens is not None
    ]

    return BenchmarkSummary(
        run_id=run_id,
        dataset_path=str(dataset_path),
        dataset_sha256=dataset_hash,
        query_count=len(results),
        success_count=len(ok),
        failure_count=len(results) - len(ok),
        model=model,
        planner_model=planner_model,
        reranker_enabled=reranker_enabled,
        started_at=started_at,
        completed_at=completed_at,
        dataset_size=dataset_size,
        selection=selection,
        performance=PerformanceSummary(
            mean_ms=mean(times),
            median_ms=percentile(times, 50),
            p95_ms=percentile(times, 95),
            min_ms=min(times) if times else 0.0,
            max_ms=max(times) if times else 0.0,
        ),
        retrieval=RetrievalSummary(
            mean_precision=mean([r.retrieval.precision for r in ok]),
            mean_recall=mean([r.retrieval.recall for r in ok]),
            hit_rate=sum(1 for r in ok if r.retrieval.hit) / len(ok) if ok else 0.0,
            full_recall_rate=sum(1 for r in ok if r.retrieval.full_recall) / len(ok)
            if ok
            else 0.0,
        ),
        quality=QualitySummary(
            mean_answer_relevancy=mean_opt([r.quality.answer_relevancy for r in ok]),
            mean_faithfulness=mean_opt([r.quality.faithfulness for r in ok]),
            mean_semantic_similarity=mean_opt(
                [r.quality.semantic_similarity for r in ok]
            ),
        ),
        tokens=TokensSummary(
            available=bool(token_totals),
            mean_total_tokens=sum(token_totals) / len(token_totals)
            if token_totals
            else None,
            total_tokens=sum(token_totals),
        ),
    )
