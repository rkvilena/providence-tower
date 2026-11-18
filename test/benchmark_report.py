"""Render a readable Markdown report for a saved benchmark run.

Reads the JSON/JSONL files produced by the benchmark runner and the optional
answer-check evaluator, then writes report.md into the same run directory.

Usage:
    python -m test.benchmark_report --run-dir test/results/baseline
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from test.benchmark_schemas import TEST_DIR, load_jsonl, now_utc_iso


def _get(data: dict, dotted: str, default: Any = None) -> Any:
    node: Any = data
    for part in dotted.split("."):
        if not isinstance(node, dict) or part not in node:
            return default
        node = node[part]
    return node


def _fmt_number(value: Any, digits: int = 3) -> str:
    if value is None:
        return "n/a"
    if isinstance(value, bool):
        return "yes" if value else "no"
    if isinstance(value, (int, float)):
        if isinstance(value, float):
            return f"{value:.{digits}f}"
        return f"{value:,}"
    return str(value)


def _fmt_ms(value: Any) -> str:
    if not isinstance(value, (int, float)):
        return "n/a"
    return f"{value:,.0f} ms"


def _fmt_pct(value: Any) -> str:
    if not isinstance(value, (int, float)):
        return "n/a"
    return f"{value * 100:.1f}%"


def _md(value: Any) -> str:
    return str(value).replace("|", "\\|").replace("\n", " ").strip()


def _preview(text: str, limit: int = 140) -> str:
    text = " ".join(text.split())
    return text if len(text) <= limit else text[: limit - 1] + "…"


def _check_summary(quality: dict | None, *, ran: bool) -> str:
    if not ran or quality is None:
        return "not run"
    status = quality.get("answer_checks_status")
    total = quality.get("checks_total", 0)
    passed = quality.get("checks_passed", 0)
    if status == "pass":
        return f"pass ({passed}/{total})"
    if status == "fail":
        return f"FAIL ({passed}/{total})"
    reason = quality.get("answer_checks_reason") or "unevaluated"
    return f"unevaluated: {reason}"


def _report_lines(
    run_dir: Path,
    summary: dict,
    rows: list[dict],
    quality_rows: list[dict],
    quality_summary: dict | None,
) -> list[str]:
    quality_by_id = {int(q["id"]): q for q in quality_rows}
    run_id = summary.get("run_id", run_dir.name)
    lines: list[str] = [
        f"# RAG Benchmark Report: {run_id}",
        "",
        f"_Generated {now_utc_iso()} from `summary.json`, `raw-results.jsonl`, "
        "and available answer-check outputs._",
        "",
        "## Run overview",
        "",
        "| Item | Value |",
        "| --- | --- |",
        f"| Run ID | {_md(run_id)} |",
        f"| Dataset | `{_md(summary.get('dataset_path', 'n/a'))}` |",
        f"| Dataset SHA256 | `{_md(str(summary.get('dataset_sha256', 'n/a'))[:16])}...` |",
        f"| Questions | {_fmt_number(summary.get('query_count'))} |",
        f"| Successful / failed | {_fmt_number(summary.get('success_count'))} / "
        f"{_fmt_number(summary.get('failure_count'))} |",
        f"| Started | {_md(summary.get('started_at') or 'n/a')} |",
        f"| Completed | {_md(summary.get('completed_at') or 'n/a')} |",
        f"| Planner model | `{_md(summary.get('planner_model') or 'n/a')}` |",
        f"| Reranker enabled | {_fmt_number(summary.get('reranker_enabled'))} |",
    ]
    selection = summary.get("selection")
    if selection:
        lines.append(f"| Selection | `{_md(json.dumps(selection))}` |")
    tokens = summary.get("tokens") or {}
    if tokens.get("available"):
        lines.append(
            f"| Tokens | {_fmt_number(tokens.get('total_tokens'))} total, "
            f"{_fmt_number(tokens.get('mean_total_tokens'), 1)} mean/query |"
        )

    metadata = summary.get("run_metadata") or {}

    def available(value: Any) -> str:
        return "unavailable" if value in (None, "") else f"`{_md(value)}`"

    dirty = metadata.get("code_dirty")
    dirty_text = "unknown" if dirty is None else ("yes" if dirty else "no")
    lines += [
        "",
        "## Reproducibility metadata",
        "",
        "| Item | Value |",
        "| --- | --- |",
        f"| Code revision | {available(metadata.get('code_revision'))} |",
        f"| Tracked working-tree changes | {dirty_text} |",
        f"| Corpus version | {available(metadata.get('corpus_version'))} |",
        f"| Index version | {available(metadata.get('index_version'))} |",
        f"| Embedding model | {available(metadata.get('embedding_model'))} |",
        f"| Query embedding model | "
        f"{available(metadata.get('query_embedding_model'))} |",
        f"| Reranker model | {available(metadata.get('reranker_model'))} |",
        f"| Generation model | {available(metadata.get('generation_model'))} |",
        "",
        "Values reported as unavailable were not discoverable for this run; "
        "they are not silently inferred.",
    ]

    performance = summary.get("performance") or {}
    retrieval = summary.get("retrieval") or {}
    lines += [
        "",
        "## Latency",
        "",
        "| Mean | Median | P95 | Min | Max |",
        "| ---: | ---: | ---: | ---: | ---: |",
        f"| {_fmt_ms(performance.get('mean_ms'))} "
        f"| {_fmt_ms(performance.get('median_ms'))} "
        f"| {_fmt_ms(performance.get('p95_ms'))} "
        f"| {_fmt_ms(performance.get('min_ms'))} "
        f"| {_fmt_ms(performance.get('max_ms'))} |",
        "",
        "## Retrieval",
        "",
        "| Precision | Recall | Hit rate | Full recall | MRR | Recall@5 |",
        "| ---: | ---: | ---: | ---: | ---: | ---: |",
        f"| {_fmt_number(retrieval.get('mean_precision'))} "
        f"| {_fmt_number(retrieval.get('mean_recall'))} "
        f"| {_fmt_number(retrieval.get('hit_rate'), 2)} "
        f"| {_fmt_number(retrieval.get('full_recall_rate'), 2)} "
        f"| {_fmt_number(retrieval.get('mean_mrr'))} "
        f"| {_fmt_number(retrieval.get('mean_recall_at_5'))} |",
        "",
        "Precision and recall use the expected chunk IDs as relevance labels. "
        "MRR and Recall@5 use the final ranked chunk list. These metrics describe "
        "retrieval, not whether the answer is correct.",
        "",
        "## Answer checks",
        "",
    ]

    if quality_summary is None:
        lines += [
            "Answer checks have not been run for this run. Run:",
            "",
            "```text",
            f"python -m test.benchmark_evaluator --run-dir {run_dir}",
            "```",
            "",
        ]
    else:
        lines += [
            "| Item | Value |",
            "| --- | ---: |",
            f"| Questions evaluated | {_fmt_number(quality_summary.get('evaluated'))} "
            f"of {_fmt_number(quality_summary.get('questions'))} |",
            f"| Unevaluated | {_fmt_number(quality_summary.get('unevaluated'))} |",
            f"| Reviewed checks passed | {_fmt_number(quality_summary.get('passed_checks'))} "
            f"of {_fmt_number(quality_summary.get('total_checks'))} |",
            f"| Check pass rate | {_fmt_pct(quality_summary.get('check_pass_rate'))} |",
            f"| Mean token overlap (diagnostic) | "
            f"{_fmt_number(quality_summary.get('mean_token_overlap'))} |",
            "",
            f"> {_md(quality_summary.get('note', ''))}",
            "",
        ]

    issues: list[str] = []
    for row in rows:
        if row.get("status") != "success":
            error = row.get("error") or {}
            issues.append(
                f"- **[{row.get('id')}] Run failed:** {error.get('type', 'Error')}: "
                f"{_preview(str(error.get('message', '')))}"
            )
            continue
        quality = quality_by_id.get(int(row["id"]))
        if quality and quality.get("answer_checks_status") == "fail":
            issues.append(
                f"- **[{row.get('id')}] Answer checks failed:** "
                f"{quality.get('checks_passed')}/{quality.get('checks_total')} passed."
            )
        elif quality and quality.get("answer_checks_status") == "unevaluated":
            issues.append(
                f"- **[{row.get('id')}] Answer checks unevaluated:** "
                f"{_md(quality.get('answer_checks_reason', 'unknown reason'))}."
            )

    lines += ["## Issues at a glance", ""]
    lines += issues or ["No run failures, failed checks, or unevaluated checks."]
    lines += [
        "",
        "## Question results",
        "",
        "| ID | Status | Latency | P | R | MRR | R@5 | Checks | Answer preview |",
        "| ---: | --- | ---: | ---: | ---: | ---: | ---: | --- | --- |",
    ]
    for row in rows:
        quality = quality_by_id.get(int(row["id"]))
        retrieval_row = row.get("retrieval") or {}
        answer = row.get("actual_answer") or ""
        lines.append(
            f"| {row.get('id')} | {_md(row.get('status', 'n/a'))} "
            f"| {_fmt_ms(row.get('total_execution_ms'))} "
            f"| {_fmt_number(retrieval_row.get('precision'), 2)} "
            f"| {_fmt_number(retrieval_row.get('recall'), 2)} "
            f"| {_fmt_number(retrieval_row.get('mrr'), 2)} "
            f"| {_fmt_number(retrieval_row.get('recall_at_5'), 2)} "
            f"| {_md(_check_summary(quality, ran=bool(quality_rows)))} "
            f"| {_md(_preview(answer))} |"
        )

    lines += ["", "## Question details", ""]
    for row in rows:
        row_id = int(row["id"])
        quality = quality_by_id.get(row_id)
        retrieval_row = row.get("retrieval") or {}
        lines += [
            f"### [{row_id}] {_md(row.get('question', ''))}",
            "",
            f"- **Status:** {_md(row.get('status', 'n/a'))}",
            f"- **Latency:** {_fmt_ms(row.get('total_execution_ms'))}",
            f"- **Expected chunks:** "
            f"`{', '.join(row.get('expected_context_chunk_ids') or []) or 'n/a'}`",
            f"- **Retrieved chunks:** "
            f"`{', '.join(row.get('retrieved_chunk_ids') or []) or 'n/a'}`",
            f"- **Retrieval:** precision {_fmt_number(retrieval_row.get('precision'))}, "
            f"recall {_fmt_number(retrieval_row.get('recall'))}, "
            f"MRR {_fmt_number(retrieval_row.get('mrr'))}, "
            f"Recall@5 {_fmt_number(retrieval_row.get('recall_at_5'))}",
        ]
        pool = row.get("candidate_chunk_ids_by_query") or {}
        if pool:
            pool_text = "; ".join(
                f"`{_md(query)}` -> {', '.join(chunk_ids)}"
                for query, chunk_ids in pool.items()
            )
            lines.append(f"- **Pre-rerank candidate pool:** {pool_text}")
        if quality:
            lines.append(
                f"- **Answer checks:** "
                f"{_md(_check_summary(quality, ran=True))}"
            )
            for check in quality.get("checks") or []:
                mark = "x" if check.get("passed") else " "
                lines.append(f"  - [{mark}] {_md(check.get('label', ''))}")
            if quality.get("token_overlap") is not None:
                lines.append(
                    f"- **Token overlap (diagnostic only):** "
                    f"{_fmt_number(quality.get('token_overlap'))}"
                )
        else:
            lines.append("- **Answer checks:** not run")

        error = row.get("error")
        if error:
            lines += [
                "",
                f"**Error:** {_md(error.get('type', 'Error'))}: "
                f"{_md(error.get('message', ''))}",
                "",
                "```text",
                str(error.get("traceback", "")).rstrip(),
                "```",
            ]

        answer = (row.get("actual_answer") or "").strip()
        lines += ["", "**Answer**", ""]
        if answer:
            lines += [f"> {line}" if line else ">" for line in answer.splitlines()]
        else:
            lines.append("_No answer returned._")
        lines.append("")

    return lines


def generate_report(run_dir: Path) -> Path:
    """Write report.md for a saved run and return its path."""
    summary_path = run_dir / "summary.json"
    raw_path = run_dir / "raw-results.jsonl"
    if not summary_path.exists():
        raise FileNotFoundError(f"Missing summary.json in run directory: {run_dir}")
    if not raw_path.exists():
        raise FileNotFoundError(f"Missing raw-results.jsonl in run directory: {run_dir}")

    with open(summary_path, "r", encoding="utf-8") as f:
        summary = json.load(f)
    rows = load_jsonl(raw_path)

    quality_path = run_dir / "quality-results.jsonl"
    quality_rows = load_jsonl(quality_path) if quality_path.exists() else []

    quality_summary_path = run_dir / "quality-summary.json"
    quality_summary = None
    if quality_summary_path.exists():
        with open(quality_summary_path, "r", encoding="utf-8") as f:
            quality_summary = json.load(f)

    report_path = run_dir / "report.md"
    report_path.write_text(
        "\n".join(_report_lines(run_dir, summary, rows, quality_rows, quality_summary))
        + "\n",
        encoding="utf-8",
    )
    return report_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Render a benchmark run report.")
    parser.add_argument("--run-dir", required=True)
    args = parser.parse_args()

    run_dir = Path(args.run_dir)
    if not run_dir.is_absolute():
        run_dir = TEST_DIR.parent / run_dir

    print(f"Report written to: {generate_report(run_dir)}")


if __name__ == "__main__":
    main()
