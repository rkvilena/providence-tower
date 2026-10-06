"""Deterministic answer-quality evaluator for saved benchmark results.

Reads a run's raw-results.jsonl, computes deterministic quality diagnostics
(required-value checks + token-overlap similarity), and writes
quality-results.jsonl. No LLM calls — fast, cheap, reproducible.

Usage:
    python -m test.benchmark_evaluator --run-dir test/results/baseline
"""
from __future__ import annotations

import argparse
import re
from pathlib import Path

from test.benchmark_schemas import TEST_DIR, load_jsonl, save_json, save_jsonl

# Normalized "significant" tokens from ground truth are required in the answer.
_STOPWORDS = frozenset(
    """a an and are as at be by for from he her his in is it its of on or that the
    their them they this to was were what which who whose with""".split()
)
_NUM_RE = re.compile(r"\d+(?:[.,]\d+)?")


def _tokens(text: str) -> set[str]:
    return {t for t in re.findall(r"[a-z0-9]+", text.lower()) if t not in _STOPWORDS}


def evaluate_answer(row: dict) -> dict:
    """Deterministic quality diagnostics for one result row."""
    answer = (row.get("actual_answer") or "").strip()
    truth = (row.get("ground_truth") or "").strip()
    out: dict = {"id": row["id"], "status": row["status"]}

    if row["status"] != "success" or not answer:
        out.update(required_values_found=0.0, token_overlap=0.0)
        return out

    answer_l = answer.lower()

    # 1. Required values: numbers, catalog codes, and significant terms from ground truth.
    numbers_in_truth = set(_NUM_RE.findall(truth))
    truth_tokens = _tokens(truth)
    missing_numbers = [n for n in numbers_in_truth if n not in answer_l]
    missing_terms = [t for t in truth_tokens if t not in answer_l]
    total_required = len(numbers_in_truth) + len(truth_tokens)
    found = total_required - len(missing_numbers) - len(missing_terms)
    out["required_values_found"] = round(found / total_required, 4) if total_required else None
    out["missing_values"] = missing_numbers + sorted(missing_terms)

    # 2. Token-overlap (bag-of-words) similarity: diagnostic only, NOT a quality verdict.
    ans_tokens = _tokens(answer)
    if ans_tokens and truth_tokens:
        out["token_overlap"] = round(len(ans_tokens & truth_tokens) / len(truth_tokens), 4)
    else:
        out["token_overlap"] = 0.0

    return out


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate saved benchmark results.")
    parser.add_argument("--run-dir", required=True)
    args = parser.parse_args()

    run_dir = Path(args.run_dir)
    if not run_dir.is_absolute():
        run_dir = TEST_DIR.parent / run_dir

    rows = load_jsonl(run_dir / "raw-results.jsonl")
    evaluated = [evaluate_answer(r) for r in rows]
    save_jsonl(run_dir / "quality-results.jsonl", evaluated)

    scored = [e for e in evaluated if e["required_values_found"] is not None]
    mean_required = (
        sum(e["required_values_found"] for e in scored) / len(scored) if scored else None
    )
    mean_overlap = (
        sum(e["token_overlap"] for e in scored) / len(scored) if scored else None
    )
    save_json(
        run_dir / "quality-summary.json",
        {
            "evaluated": len(evaluated),
            "mean_required_values_found": mean_required,
            "mean_token_overlap": mean_overlap,
            "note": "token_overlap is a diagnostic; LLM judging (faithfulness/relevancy) pending",
        },
    )

    print(f"Evaluated {len(evaluated)} results -> {run_dir / 'quality-results.jsonl'}")
    if mean_required is not None:
        print(f"Mean required-values coverage: {mean_required:.3f}")
        print(f"Mean token overlap:            {mean_overlap:.3f}")


if __name__ == "__main__":
    main()
