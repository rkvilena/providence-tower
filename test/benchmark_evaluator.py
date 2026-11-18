"""Deterministic answer-check evaluator for saved benchmark results.

Reads a run's raw-results.jsonl and the golden dataset's reviewed
``answer_checks`` criteria, then writes quality-results.jsonl. No LLM calls:
fast, cheap, reproducible.

The checks cover only explicit facts or values chosen for a question. They are
not a complete verdict on answer quality. Questions without reviewed checks
are reported as unevaluated.

Usage:
    python -m test.benchmark_evaluator --run-dir test/results/baseline
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path
from typing import Any

from test.benchmark_report import generate_report
from test.benchmark_schemas import TEST_DIR, load_jsonl, save_json, save_jsonl

GOLDEN_DATASET = TEST_DIR / "GOLDEN-DATASET.jsonl"
_TOKEN_RE = re.compile(r"[a-z0-9]+")
_DIGIT_COMMA_RE = re.compile(r"(?<=\d),(?=\d)")

# Token overlap is retained only as a diagnostic signal, never as a verdict.
_STOPWORDS = frozenset(
    """a an and are as at be by for from he her his in is it its of on or that the
    their them they this to was were what which who whose with""".split()
)


def _tokens(text: str) -> set[str]:
    return {t for t in _TOKEN_RE.findall(text.lower()) if t not in _STOPWORDS}


def _normalize(text: str) -> str:
    text = text.casefold().replace("\u2019", "'")
    text = _DIGIT_COMMA_RE.sub("", text)
    return " ".join(re.findall(r"[a-z0-9]+", text))


def _contains_term(answer_norm: str, term: str) -> bool:
    term_norm = _normalize(term)
    if not term_norm:
        return False
    pattern = rf"(?<![a-z0-9]){re.escape(term_norm)}(?![a-z0-9])"
    return re.search(pattern, answer_norm) is not None


def _check_result(answer_norm: str, check: dict[str, Any]) -> dict[str, Any]:
    """Evaluate one reviewed check, returning its label and pass state."""
    if "any_of" in check:
        terms = check["any_of"]
        passed = any(_contains_term(answer_norm, term) for term in terms)
    else:
        terms = check["all_of"]
        passed = all(_contains_term(answer_norm, term) for term in terms)
    return {"label": check["label"], "passed": passed}


def evaluate_answer(row: dict, checks: list[dict] | None = None) -> dict:
    """Evaluate one result row against its reviewed answer checks."""
    checks = checks or []
    answer = (row.get("actual_answer") or "").strip()
    truth = (row.get("ground_truth") or "").strip()
    out: dict = {
        "id": row["id"],
        "status": row["status"],
        "checks_total": len(checks),
        "checks_passed": 0,
    }

    answer_norm = _normalize(answer)
    evaluated = row["status"] == "success" and bool(answer) and bool(checks)
    if not evaluated:
        if row["status"] != "success":
            reason = "question run failed"
        elif not answer:
            reason = "answer was empty"
        else:
            reason = "no reviewed answer checks"
        out.update(
            answer_checks_status="unevaluated",
            answer_checks_reason=reason,
            checks=[],
        )
    else:
        check_results = [_check_result(answer_norm, check) for check in checks]
        passed = sum(1 for result in check_results if result["passed"])
        out.update(
            answer_checks_status="pass" if passed == len(checks) else "fail",
            answer_checks_reason=None,
            checks=check_results,
            checks_passed=passed,
        )

    # Diagnostic only: bag-of-words overlap with the reference answer.
    ans_tokens = _tokens(answer)
    truth_tokens = _tokens(truth)
    out["token_overlap"] = (
        round(len(ans_tokens & truth_tokens) / len(truth_tokens), 4)
        if ans_tokens and truth_tokens
        else None
    )
    return out


def _validate_checks(checks: list[dict], row_id: int) -> None:
    for check in checks:
        if not isinstance(check, dict) or not check.get("label"):
            raise ValueError(f"Dataset question {row_id} has a check without a label.")
        has_any = isinstance(check.get("any_of"), list) and bool(check["any_of"])
        has_all = isinstance(check.get("all_of"), list) and bool(check["all_of"])
        if has_any == has_all:
            raise ValueError(
                f"Dataset question {row_id} check {check['label']!r} must have "
                "exactly one non-empty any_of or all_of list."
            )


def load_answer_checks(path: Path) -> dict[int, list[dict]]:
    """Load and validate reviewed answer checks, keyed by question ID."""
    checks_by_id: dict[int, list[dict]] = {}
    for row in load_jsonl(path):
        row_id = int(row["id"])
        checks = row.get("answer_checks", [])
        if not isinstance(checks, list):
            raise ValueError(f"Dataset question {row_id} answer_checks must be a list.")
        _validate_checks(checks, row_id)
        checks_by_id[row_id] = checks
    return checks_by_id


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate saved benchmark results.")
    parser.add_argument("--run-dir", required=True)
    parser.add_argument(
        "--dataset",
        default=str(GOLDEN_DATASET),
        help="Dataset containing reviewed answer_checks (default: golden dataset).",
    )
    args = parser.parse_args()

    run_dir = Path(args.run_dir)
    if not run_dir.is_absolute():
        run_dir = TEST_DIR.parent / run_dir

    dataset_path = Path(args.dataset)
    if not dataset_path.is_absolute():
        dataset_path = TEST_DIR.parent / dataset_path

    checks_by_id = load_answer_checks(dataset_path)
    rows = load_jsonl(run_dir / "raw-results.jsonl")
    evaluated = [
        evaluate_answer(row, checks_by_id.get(int(row["id"]), [])) for row in rows
    ]
    save_jsonl(run_dir / "quality-results.jsonl", evaluated)

    with_checks = [e for e in evaluated if e["answer_checks_status"] != "unevaluated"]
    total_checks = sum(e["checks_total"] for e in with_checks)
    passed_checks = sum(e["checks_passed"] for e in with_checks)
    overlaps = [e["token_overlap"] for e in evaluated if e["token_overlap"] is not None]
    save_json(
        run_dir / "quality-summary.json",
        {
            "questions": len(evaluated),
            "evaluated": len(with_checks),
            "unevaluated": len(evaluated) - len(with_checks),
            "total_checks": total_checks,
            "passed_checks": passed_checks,
            "check_pass_rate": round(passed_checks / total_checks, 4)
            if total_checks
            else None,
            "mean_token_overlap": round(sum(overlaps) / len(overlaps), 4)
            if overlaps
            else None,
            "note": (
                "answer_checks cover only reviewed facts/values and are not a "
                "complete quality verdict; token_overlap is a diagnostic only."
            ),
        },
    )

    print(
        f"Evaluated {len(with_checks)}/{len(evaluated)} results "
        f"({len(evaluated) - len(with_checks)} unevaluated) -> "
        f"{run_dir / 'quality-results.jsonl'}"
    )
    if total_checks:
        print(
            f"Reviewed checks passed: {passed_checks}/{total_checks} "
            f"({passed_checks / total_checks:.1%})"
        )
    if overlaps:
        print(f"Mean token overlap (diagnostic): {sum(overlaps) / len(overlaps):.3f}")
    if (run_dir / "summary.json").exists():
        print(f"Report refreshed: {generate_report(run_dir)}")


if __name__ == "__main__":
    main()
