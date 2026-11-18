"""Run dataset validation, the RAG benchmark, and answer-check evaluation."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
_RESULTS_PREFIX = "Results saved to:"


def _run_directory(stdout: str) -> str | None:
    for line in stdout.splitlines():
        if line.startswith(_RESULTS_PREFIX):
            path = line[len(_RESULTS_PREFIX) :].strip()
            return path or None
    return None


def _forward_output(stdout: str, stderr: str) -> None:
    sys.stdout.write(stdout)
    sys.stderr.write(stderr)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Check the golden dataset, run the RAG benchmark, and evaluate "
            "saved answers."
        )
    )
    parser.add_argument("--run-id", help="Optional label for this benchmark run.")
    selection = parser.add_mutually_exclusive_group()
    selection.add_argument(
        "--ids",
        help="Comma-separated question IDs to run (e.g. --ids 1,10,12).",
    )
    selection.add_argument(
        "--limit",
        type=int,
        help="Run only the first N dataset questions.",
    )
    parser.add_argument(
        "--warmup",
        action="store_true",
        help="Send one warm-up query before measuring (result discarded).",
    )
    parser.add_argument(
        "--output-dir",
        help="Directory for this run's results (default: test/results/<run_id>).",
    )
    args = parser.parse_args(argv)

    check = subprocess.run(
        [sys.executable, "test/sync_golden_dataset.py", "--check"],
        cwd=PROJECT_ROOT,
        check=False,
    )
    if check.returncode:
        return check.returncode

    runner = [sys.executable, "-m", "test.benchmark_runner"]
    if args.run_id is not None:
        runner.extend(["--run-id", args.run_id])
    if args.ids is not None:
        runner.extend(["--ids", args.ids])
    if args.limit is not None:
        runner.extend(["--limit", str(args.limit)])
    if args.warmup:
        runner.append("--warmup")
    if args.output_dir is not None:
        runner.extend(["--output-dir", args.output_dir])

    benchmark = subprocess.run(
        runner,
        cwd=PROJECT_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    _forward_output(benchmark.stdout, benchmark.stderr)
    if benchmark.returncode:
        return benchmark.returncode

    run_dir = _run_directory(benchmark.stdout)
    if run_dir is None:
        print(
            "Could not determine the benchmark run directory from runner output.",
            file=sys.stderr,
        )
        return 1

    evaluation = subprocess.run(
        [
            sys.executable,
            "-m",
            "test.benchmark_evaluator",
            "--run-dir",
            run_dir,
        ],
        cwd=PROJECT_ROOT,
        check=False,
    )
    return evaluation.returncode


if __name__ == "__main__":
    raise SystemExit(main())
