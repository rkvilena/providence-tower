"""Unit tests for the readable golden dataset source and JSONL synchronizer."""

from __future__ import annotations

import json

from test.sync_golden_dataset import (
    JSONL_OUTPUT,
    JSON_SOURCE,
    is_current,
    load_records,
    main,
    render_jsonl,
    write_jsonl,
)


def test_json_source_round_trips_into_jsonl_with_same_content_and_order():
    records = load_records(JSON_SOURCE)

    generated = [json.loads(line) for line in render_jsonl(records).splitlines()]

    assert len(records) == 20
    assert generated == records
    assert [record["id"] for record in records] == list(range(1, 21))
    assert [record["id"] for record in generated] == [record["id"] for record in records]


def test_generated_output_is_deterministic(tmp_path):
    records = load_records(JSON_SOURCE)
    first = tmp_path / "first.jsonl"
    second = tmp_path / "second.jsonl"

    write_jsonl(records, first)
    write_jsonl(records, second)

    assert render_jsonl(records) == render_jsonl(records)
    assert first.read_bytes() == second.read_bytes()


def test_checked_in_jsonl_matches_the_json_source():
    assert is_current(load_records(JSON_SOURCE), JSONL_OUTPUT) is True


def test_check_mode_reports_stale_output_without_rewriting(tmp_path, capsys):
    records = load_records(JSON_SOURCE)
    stale = tmp_path / "stale.jsonl"
    stale.write_text('{"id": 999}\n', encoding="utf-8")

    exit_code = main(["--check", "--output", str(stale)])

    assert exit_code == 1
    assert "stale" in capsys.readouterr().out
    assert stale.read_text(encoding="utf-8") == '{"id": 999}\n'
    assert is_current(records, stale) is False


def test_check_mode_reports_missing_output(tmp_path, capsys):
    missing = tmp_path / "missing.jsonl"

    exit_code = main(["--check", "--output", str(missing)])

    assert exit_code == 1
    assert "missing" in capsys.readouterr().out
    assert not missing.exists()


def test_write_mode_regenerates_output(tmp_path, capsys):
    records = load_records(JSON_SOURCE)
    output = tmp_path / "generated.jsonl"

    exit_code = main(["--output", str(output)])

    assert exit_code == 0
    assert f"wrote {len(records)} records" in capsys.readouterr().out
    assert output.read_text(encoding="utf-8") == render_jsonl(records)
    assert is_current(records, output) is True
