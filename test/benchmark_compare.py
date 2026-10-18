"""Compare two benchmark runs (baseline vs candidate) from saved result dirs.

Usage:
    python -m test.benchmark_compare --baseline test/results/baseline --candidate test/results/improved
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from test.benchmark_schemas import TEST_DIR, load_jsonl

# metric label -> (json path in summary, lower_is_better)
METRICS: list[tuple[str, str, bool]] = [
    ("Median execution (ms)", "performance.median_ms", True),
    ("Mean execution (ms)", "performance.mean_ms", True),
    ("P95 execution (ms)", "performance.p95_ms", True),
    ("Mean context precision", "retrieval.mean_precision", False),
    ("Mean context recall", "retrieval.mean_recall", False),
    ("Hit rate", "retrieval.hit_rate", False),
    ("Full recall rate", "retrieval.full_recall_rate", False),
    ("Mean answer relevancy", "quality.mean_answer_relevancy", False),
    ("Mean faithfulness", "quality.mean_faithfulness", False),
    ("Total tokens", "tokens.total_tokens", True),
]


def _get(data: dict, dotted: str):
    node = data
    for part in dotted.split("."):
        if not isinstance(node, dict) or part not in node:
            return None
        node = node[part]
    return node


def _fmt(v) -> str:
    if v is None:
        return "n/a"
    if isinstance(v, float):
        return f"{v:.3f}" if abs(v) < 100 else f"{v:,.0f}"
    return f"{v:,}"


def load_run(path: Path) -> dict:
    summary_file = path / "summary.json"
    if not summary_file.exists():
        raise FileNotFoundError(f"Not a run directory (missing summary.json): {path}")
    with open(summary_file, "r", encoding="utf-8") as f:
        return json.load(f)


def compare(baseline: dict, candidate: dict) -> None:
    run_a = baseline.get("run_id", "?")
    run_b = candidate.get("run_id", "?")
    print(f"Baseline : {run_a}")
    print(f"Candidate: {run_b}")
    print()
    print(f"{'Metric':<26} {'Baseline':>12} {'Candidate':>12} {'Change':>12}  Verdict")
    print("-" * 78)

    regressions: list[str] = []
    for label, dotted, lower_is_better in METRICS:
        a = _get(baseline, dotted)
        b = _get(candidate, dotted)
        if a in (None, 0) or b is None:
            print(
                f"{label:<26} {_fmt(a):>12} {_fmt(b):>12} {'—':>12}  (not comparable)"
            )
            continue
        diff = b - a
        pct = diff / a * 100
        better = diff < 0 if lower_is_better else diff > 0
        worse = diff > 0 if lower_is_better else diff < 0
        verdict = "improved" if better else ("REGRESSED" if worse else "unchanged")
        if worse:
            regressions.append(label)
        print(f"{label:<26} {_fmt(a):>12} {_fmt(b):>12} {pct:>+11.1f}%  {verdict}")

    print()
    if regressions:
        print("Regressions detected in: " + ", ".join(regressions))
    else:
        print("No regressions detected.")


def compare_per_question(baseline_dir: Path, candidate_dir: Path) -> None:
    """Show per-question latency and recall changes between two runs."""
    base_rows = {r["id"]: r for r in load_jsonl(baseline_dir / "raw-results.jsonl")}
    cand_rows = {r["id"]: r for r in load_jsonl(candidate_dir / "raw-results.jsonl")}

    print()
    print("Per-question changes (id | latency ms base->cand | recall base->cand):")
    print("-" * 60)
    for qid in sorted(set(base_rows) & set(cand_rows)):
        b, c = base_rows[qid], cand_rows[qid]
        print(
            f"[{qid:>3}] {b['total_execution_ms']:>8.0f} -> {c['total_execution_ms']:>8.0f} ms | "
            f"{b['retrieval']['recall']:.2f} -> {c['retrieval']['recall']:.2f}"
            + ("  (ERROR)" if c["status"] == "error" else "")
        )


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare two benchmark runs.")
    parser.add_argument("--baseline", required=True)
    parser.add_argument("--candidate", required=True)
    parser.add_argument("--per-question", action="store_true")
    args = parser.parse_args()

    baseline_dir = Path(args.baseline)
    candidate_dir = Path(args.candidate)
    if not baseline_dir.is_absolute():
        baseline_dir = TEST_DIR.parent / baseline_dir
    if not candidate_dir.is_absolute():
        candidate_dir = TEST_DIR.parent / candidate_dir

    compare(load_run(baseline_dir), load_run(candidate_dir))
    if args.per_question:
        compare_per_question(baseline_dir, candidate_dir)


if __name__ == "__main__":
    main()
