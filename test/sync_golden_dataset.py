"""Synchronize the readable golden dataset JSON with its generated JSONL file.

`test/GOLDEN-DATASET.json` is the hand-edited source of truth. The benchmark
runner and evaluator read `test/GOLDEN-DATASET.jsonl`, which is generated from
the JSON source and checked in.

Usage:
    python test/sync_golden_dataset.py
    python test/sync_golden_dataset.py --check
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

DATASET_DIR = Path(__file__).resolve().parent
JSON_SOURCE = DATASET_DIR / "GOLDEN-DATASET.json"
JSONL_OUTPUT = DATASET_DIR / "GOLDEN-DATASET.jsonl"


def load_records(source: Path = JSON_SOURCE) -> list[dict]:
    """Load the canonical records from the readable JSON source."""
    with source.open(encoding="utf-8") as handle:
        records = json.load(handle)
    if not isinstance(records, list):
        raise ValueError(f"{source} must contain a top-level JSON array")
    return records


def render_jsonl(records: list[dict]) -> str:
    """Render one compact JSON record per line in the given record order."""
    return "".join(json.dumps(record, ensure_ascii=False) + "\n" for record in records)


def write_jsonl(records: list[dict], output: Path = JSONL_OUTPUT) -> None:
    """Write the generated JSONL file with deterministic UTF-8/LF bytes."""
    with output.open("w", encoding="utf-8", newline="\n") as handle:
        handle.write(render_jsonl(records))


def is_current(records: list[dict], output: Path = JSONL_OUTPUT) -> bool:
    """Return whether the generated JSONL file exists and matches the source."""
    if not output.exists():
        return False
    return output.read_text(encoding="utf-8") == render_jsonl(records)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Regenerate test/GOLDEN-DATASET.jsonl from test/GOLDEN-DATASET.json."
    )
    parser.add_argument(
        "--check",
        action="store_true",
        help="report a stale or missing JSONL without rewriting it",
    )
    parser.add_argument("--source", type=Path, default=JSON_SOURCE)
    parser.add_argument("--output", type=Path, default=JSONL_OUTPUT)
    args = parser.parse_args(argv)

    records = load_records(args.source)
    if args.check:
        if is_current(records, args.output):
            print(f"{args.output} is up to date")
            return 0
        if not args.output.exists():
            print(f"{args.output} is missing; run: python test/sync_golden_dataset.py")
        else:
            print(f"{args.output} is stale; run: python test/sync_golden_dataset.py")
        return 1

    write_jsonl(records, args.output)
    print(f"wrote {len(records)} records to {args.output}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
