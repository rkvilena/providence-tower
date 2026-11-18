"""Unit tests for the one-command benchmark orchestration."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path
from unittest.mock import call, patch

from test import run_benchmark

PROJECT_ROOT = Path(run_benchmark.__file__).resolve().parent.parent
RUN_DIR = str(PROJECT_ROOT / "test" / "results" / "run-20261118T120000")


def completed(returncode: int = 0, stdout: str = "", stderr: str = ""):
    return subprocess.CompletedProcess([], returncode, stdout=stdout, stderr=stderr)


@patch("test.run_benchmark.subprocess.run")
def test_workflow_order_options_and_runner_default_run_id(mock_run):
    mock_run.side_effect = [
        completed(),
        completed(stdout=f"Results saved to: {RUN_DIR}\n"),
        completed(),
    ]

    result = run_benchmark.main(
        ["--ids", "2,4", "--warmup", "--output-dir", "custom/results"]
    )

    assert result == 0
    assert mock_run.call_args_list == [
        call(
            [sys.executable, "test/sync_golden_dataset.py", "--check"],
            cwd=PROJECT_ROOT,
            check=False,
        ),
        call(
            [
                sys.executable,
                "-m",
                "test.benchmark_runner",
                "--ids",
                "2,4",
                "--warmup",
                "--output-dir",
                "custom/results",
            ],
            cwd=PROJECT_ROOT,
            capture_output=True,
            text=True,
            check=False,
        ),
        call(
            [
                sys.executable,
                "-m",
                "test.benchmark_evaluator",
                "--run-dir",
                RUN_DIR,
            ],
            cwd=PROJECT_ROOT,
            check=False,
        ),
    ]


@patch("test.run_benchmark.subprocess.run")
def test_workflow_forwards_run_id_and_limit(mock_run):
    mock_run.side_effect = [
        completed(),
        completed(stdout=f"Results saved to: {RUN_DIR}\n"),
        completed(),
    ]

    assert run_benchmark.main(["--run-id", "smoke", "--limit", "3"]) == 0
    assert mock_run.call_args_list[1].args[0][-4:] == [
        "--run-id",
        "smoke",
        "--limit",
        "3",
    ]


@patch("test.run_benchmark.subprocess.run")
def test_workflow_stops_on_each_failed_stage(mock_run):
    for failed_stage in range(3):
        mock_run.reset_mock()
        mock_run.side_effect = None
        mock_run.side_effect = [
            completed(11) if failed_stage == 0 else completed(),
            completed(12, stdout=f"Results saved to: {RUN_DIR}\n")
            if failed_stage == 1
            else completed(stdout=f"Results saved to: {RUN_DIR}\n"),
            completed(13) if failed_stage == 2 else completed(),
        ]

        result = run_benchmark.main([])

        assert result == (11, 12, 13)[failed_stage]
        assert mock_run.call_count == failed_stage + 1
