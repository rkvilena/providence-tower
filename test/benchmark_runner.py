"""Benchmark runner: executes the golden dataset through RagService and saves raw results.

Usage (from project root):
    python -m test.benchmark_runner --run-id baseline
"""

from __future__ import annotations

import argparse
import subprocess
import time
import traceback

from core.api.service import RagService
from core.env import settings
from core.inference.cloudflare_workers_ai import (
    CLOUDFLARE_EMBEDDING_MODEL_ID,
    CLOUDFLARE_RERANK_MODEL_ID,
)

from test.benchmark_report import generate_report
from test.benchmark_schemas import (
    TEST_DIR,
    BenchmarkResult,
    RetrievalMetrics,
    RunMetadata,
    file_sha256,
    load_jsonl,
    now_utc_iso,
    parse_expected_chunk_ids,
    save_json,
    save_jsonl,
    summarize,
)

GOLDEN_DATASET = TEST_DIR / "GOLDEN-DATASET.jsonl"
REQUIRED_FIELDS = {"id", "question", "context_chunk", "ground_truth"}


def load_dataset(path) -> list[dict]:
    rows = load_jsonl(path)
    for i, row in enumerate(rows):
        missing = REQUIRED_FIELDS - row.keys()
        if missing:
            raise ValueError(f"Dataset row {i} missing fields: {sorted(missing)}")
    return rows


def _git_output(*args: str) -> str | None:
    try:
        completed = subprocess.run(
            ["git", *args],
            cwd=TEST_DIR.parent,
            capture_output=True,
            text=True,
            check=True,
            timeout=5,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    return completed.stdout.strip()


def collect_run_metadata() -> RunMetadata:
    """Collect available, non-secret run-identifying details."""
    revision = _git_output("rev-parse", "HEAD")
    status = _git_output("status", "--porcelain", "--untracked-files=no")
    return RunMetadata(
        code_revision=revision,
        code_dirty=None if status is None else bool(status),
        corpus_version=None,
        index_version=None,
        embedding_model=settings.EMBEDDING_MODEL or None,
        query_embedding_model=CLOUDFLARE_EMBEDDING_MODEL_ID,
        reranker_model=CLOUDFLARE_RERANK_MODEL_ID if settings.RERANK_ENABLED else None,
        generation_model=settings.THINKER_MODEL or None,
    )


def compute_retrieval_metrics(
    expected: list[str], retrieved: list[str]
) -> RetrievalMetrics:
    exp_set, ret_set = set(expected), set(retrieved)
    intersection = exp_set & ret_set
    precision = len(intersection) / len(ret_set) if ret_set else 0.0
    recall = len(intersection) / len(exp_set) if exp_set else 0.0
    hit = bool(intersection)
    full_recall = bool(exp_set) and exp_set <= ret_set

    # Ranking metrics use the first occurrence of each retrieved chunk ID.
    ranked_unique = list(dict.fromkeys(retrieved))
    first_relevant_rank = next(
        (rank for rank, chunk_id in enumerate(ranked_unique, start=1)
         if chunk_id in exp_set),
        None,
    )
    mrr = 1.0 / first_relevant_rank if first_relevant_rank else 0.0
    recall_at_5 = (
        len(exp_set & set(ranked_unique[:5])) / len(exp_set) if exp_set else 0.0
    )

    return RetrievalMetrics(
        precision=round(precision, 4),
        recall=round(recall, 4),
        hit=hit,
        full_recall=full_recall,
        mrr=round(mrr, 4),
        recall_at_5=round(recall_at_5, 4),
    )


def run_question(service: RagService, row: dict) -> BenchmarkResult:
    expected_ids = parse_expected_chunk_ids(row["context_chunk"])
    result = BenchmarkResult(
        id=int(row["id"]),
        question=row["question"],
        expected_context_chunk_ids=expected_ids,
        ground_truth=row["ground_truth"],
    )

    start = time.perf_counter()
    try:
        state = service.run_rag(row["question"])
        result.total_execution_ms = round((time.perf_counter() - start) * 1000, 2)
        result.actual_answer = state.thinker_state.response
        result.retrieved_chunk_ids = [c.chunk_id for c in state.fetcher_state.chunks]
        result.candidate_chunk_ids_by_query = dict(
            state.fetcher_state.candidate_chunk_ids_by_query
        )
        result.used_chunk_ids = list(state.thinker_state.used_chunk_ids)
        result.node_latencies_ms = dict(state.node_latencies_ms)
        fs = state.fetcher_state
        if fs is not None and (fs.query_timings or fs.totals is not None):
            result.fetcher_timing = {
                "queries": [t.model_dump() for t in fs.query_timings],
                "totals": fs.totals.model_dump() if fs.totals else None,
            }
        if state.thinker_timing is not None:
            result.thinker_timing = state.thinker_timing.model_dump()
        result.used_chunk_ids = list(state.thinker_state.used_chunk_ids)
        result.retrieval = compute_retrieval_metrics(
            expected_ids, result.retrieved_chunk_ids
        )
        used_set = set(result.used_chunk_ids)
        exp_set = set(expected_ids)
        result.used_chunk_recall = (
            round(len(exp_set & used_set) / len(exp_set), 4) if exp_set else 0.0
        )
    except Exception as exc:  # keep the run alive; record and continue
        result.total_execution_ms = round((time.perf_counter() - start) * 1000, 2)
        result.status = "error"
        result.error = {
            "type": type(exc).__name__,
            "message": str(exc),
            "traceback": traceback.format_exc(limit=5),
        }
    return result


def select_rows(
    rows: list[dict], *, ids: list[int] | None = None, limit: int | None = None
) -> tuple[list[dict], dict]:
    """Deterministically select a subset of dataset rows.

    Returns the selected rows (in dataset order) and a selection descriptor
    recorded in summary.json. Dataset order is always preserved regardless of
    the order the IDs were given.
    """
    available = {int(r["id"]) for r in rows}
    if ids is not None:
        if not ids:
            raise ValueError("--ids was given but produced an empty list.")
        duplicates = sorted({i for i in ids if ids.count(i) > 1})
        if duplicates:
            raise ValueError(f"--ids contains duplicate values: {duplicates}")
        unknown = sorted(set(ids) - available)
        if unknown:
            raise ValueError(
                f"--ids contains IDs not present in the dataset: {unknown} "
                f"(available: {sorted(available)})"
            )
        wanted = set(ids)
        selected = [r for r in rows if int(r["id"]) in wanted]
        return selected, {"mode": "ids", "ids": sorted(set(ids))}

    if limit is not None:
        if limit < 1:
            raise ValueError("--limit must be >= 1.")
        if limit > len(rows):
            raise ValueError(f"--limit ({limit}) exceeds dataset size ({len(rows)}).")
        return rows[:limit], {
            "mode": "limit",
            "limit": limit,
            "ids": [int(r["id"]) for r in rows[:limit]],
        }

    return rows, {"mode": "all", "ids": [int(r["id"]) for r in rows]}


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run the golden-dataset RAG benchmark."
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Directory for this run's results (default: test/results/<run_id>).",
    )
    parser.add_argument(
        "--run-id",
        default=None,
        help="Label for this run (default: run-<timestamp>).",
    )
    parser.add_argument(
        "--warmup",
        action="store_true",
        help="Send one warm-up query before measuring (result discarded).",
    )
    parser.add_argument(
        "--ids",
        default=None,
        help="Comma-separated question IDs to run (e.g. --ids 1,10,12). "
        "Cannot be combined with --limit.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Run only the first N dataset questions. Cannot be combined with --ids.",
    )
    args = parser.parse_args()

    if args.ids is not None and args.limit is not None:
        parser.error("--ids and --limit are mutually exclusive.")

    requested_ids: list[int] | None = None
    if args.ids is not None:
        try:
            requested_ids = [int(part) for part in args.ids.split(",") if part.strip()]
        except ValueError:
            parser.error(
                f"--ids must be a comma-separated list of integers, got: {args.ids!r}"
            )

    run_id = (
        args.run_id or f"run-{now_utc_iso().replace(':', '').replace('-', '')[:15]}"
    )
    output_dir = (
        TEST_DIR / "results" / run_id
        if not args.output_dir
        else TEST_DIR.parent / args.output_dir
    )

    dataset_path = GOLDEN_DATASET
    dataset_hash = file_sha256(dataset_path)
    all_rows = load_dataset(dataset_path)
    rows, selection = select_rows(all_rows, ids=requested_ids, limit=args.limit)

    print(f"Run ID:            {run_id}")
    print(f"Dataset:           {dataset_path} ({len(all_rows)} questions)")
    print(f"Dataset SHA256:    {dataset_hash[:16]}...")
    sel_desc = (
        f"ids={selection['ids']} ({len(rows)}/{len(all_rows)} questions)"
        if selection["mode"] != "all"
        else f"all ({len(rows)}/{len(all_rows)} questions)"
    )
    print(f"Selection:         {sel_desc}")
    print(
        f"Planner LLM:       {'ON' if settings.PLANNER_AGENT and settings.OPENAI_API_KEY else 'OFF'}"
    )
    print(f"Planner mode:      {settings.PLANNER_MODE}")
    print(f"Planner model:     {settings.PLANNER_MODEL}")
    print(f"Reranker:          {'ON' if settings.RERANK_ENABLED else 'OFF'}")
    run_metadata = collect_run_metadata()
    revision = run_metadata.code_revision or "unavailable"
    dirty = (
        " (dirty working tree)"
        if run_metadata.code_dirty is True
        else ""
    )
    print(f"Code revision:     {revision[:12]}{dirty}")
    print()

    service = RagService()
    service.startup()

    if args.warmup:
        print("Warm-up query (discarded)...")
        run_question(
            service,
            {
                "id": 0,
                "question": "Who is Adol Christin?",
                "context_chunk": "",
                "ground_truth": "",
            },
        )
        print("Warm-up complete.")
        print()

    started_at = now_utc_iso()
    results: list[BenchmarkResult] = []
    for row in rows:
        print(f"[{row['id']:>2}] {row['question'][:70]}...", flush=True)
        result = run_question(service, row)
        results.append(result)
        if result.status == "success":
            print(
                f"     ok   {result.total_execution_ms:>9.0f} ms | "
                f"recall={result.retrieval.recall:.2f} | "
                f"chunks={len(result.retrieved_chunk_ids)}"
            )
        else:
            print(f"     ERROR: {result.error['type']}: {result.error['message'][:80]}")
    completed_at = now_utc_iso()

    service.shutdown()
    write_outputs(
        output_dir,
        run_id,
        dataset_path,
        dataset_hash,
        results,
        started_at,
        completed_at,
        selection=selection,
        dataset_size=len(all_rows),
        run_metadata=run_metadata,
    )
    print(f"Report: {generate_report(output_dir)}")


def write_outputs(
    output_dir,
    run_id,
    dataset_path,
    dataset_hash,
    results,
    started_at,
    completed_at,
    *,
    selection: dict | None = None,
    dataset_size: int = 0,
    run_metadata: RunMetadata | None = None,
) -> None:
    summary = summarize(
        run_id=run_id,
        dataset_path=dataset_path,
        dataset_hash=dataset_hash,
        results=results,
        model=settings.PLANNER_MODEL,
        planner_model=settings.PLANNER_MODEL,
        reranker_enabled=settings.RERANK_ENABLED,
        started_at=started_at,
        completed_at=completed_at,
        selection=selection,
        dataset_size=dataset_size,
        run_metadata=run_metadata,
    )

    save_jsonl(output_dir / "raw-results.jsonl", [r.to_dict() for r in results])
    save_json(output_dir / "summary.json", summary.to_dict())

    p = summary.performance
    ret = summary.retrieval
    print()
    print("=" * 60)
    print(f"Results saved to: {output_dir}")
    print(
        f"Success: {summary.success_count}/{summary.query_count}  Failures: {summary.failure_count}"
    )
    print(
        f"Latency  mean={p.mean_ms:.0f}ms  median={p.median_ms:.0f}ms  p95={p.p95_ms:.0f}ms"
    )
    print(
        f"Retrieval precision={ret.mean_precision:.3f}  recall={ret.mean_recall:.3f}  "
        f"hit_rate={ret.hit_rate:.2f}  full_recall={ret.full_recall_rate:.2f}"
    )
    print(
        f"Ranking   MRR={ret.mean_mrr:.3f}  Recall@5={ret.mean_recall_at_5:.3f}"
    )


if __name__ == "__main__":
    main()
