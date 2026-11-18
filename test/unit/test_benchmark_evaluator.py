"""Unit tests for reviewed deterministic answer checks."""

from __future__ import annotations

from test.benchmark_evaluator import (
    GOLDEN_DATASET,
    evaluate_answer,
    load_answer_checks,
)


def make_row(answer: str, status: str = "success") -> dict:
    return {
        "id": 1,
        "status": status,
        "actual_answer": answer,
        "ground_truth": "The answer mentions a Silver Sword and Claws.",
    }


def test_any_of_accepts_a_reviewed_alias_case_and_punctuation_insensitively():
    checks = [{"label": "names the Silver Sword", "any_of": ["Silver Sword"]}]

    result = evaluate_answer(make_row("He exchanged the silver-sword."), checks)

    assert result["answer_checks_status"] == "pass"
    assert result["checks_passed"] == 1
    assert result["checks"][0]["passed"] is True


def test_all_of_requires_every_listed_fact():
    checks = [{"label": "lists the three colors", "all_of": ["red", "blue", "gold"]}]

    passed = evaluate_answer(make_row("Red, blue, and gold Emelas."), checks)
    failed = evaluate_answer(make_row("Red and blue Emelas."), checks)

    assert passed["answer_checks_status"] == "pass"
    assert failed["answer_checks_status"] == "fail"
    assert failed["checks_passed"] == 0


def test_failed_check_is_reported_even_when_other_checks_pass():
    checks = [
        {"label": "names the Silver Sword", "any_of": ["Silver Sword"]},
        {"label": "names Claws", "any_of": ["Claws"]},
    ]

    result = evaluate_answer(make_row("He kept the Silver Sword."), checks)

    assert result["answer_checks_status"] == "fail"
    assert result["checks_total"] == 2
    assert result["checks_passed"] == 1
    assert result["checks"][1]["passed"] is False


def test_question_without_reviewed_checks_is_unevaluated():
    result = evaluate_answer(make_row("An answer."), [])

    assert result["answer_checks_status"] == "unevaluated"
    assert result["answer_checks_reason"] == "no reviewed answer checks"
    assert result["checks"] == []
    assert result["token_overlap"] is not None


def test_failed_run_or_empty_answer_is_unevaluated():
    checks = [{"label": "names Claws", "any_of": ["Claws"]}]

    failed_run = evaluate_answer(make_row("", status="error"), checks)
    empty_answer = evaluate_answer(make_row(""), checks)

    assert failed_run["answer_checks_status"] == "unevaluated"
    assert failed_run["answer_checks_reason"] == "question run failed"
    assert empty_answer["answer_checks_status"] == "unevaluated"
    assert empty_answer["answer_checks_reason"] == "answer was empty"


def test_golden_dataset_has_valid_reviewed_checks_for_every_question():
    checks_by_id = load_answer_checks(GOLDEN_DATASET)

    assert len(checks_by_id) == 20
    assert all(checks for checks in checks_by_id.values())


def test_question_16_keeps_each_def_value_with_its_item():
    checks = load_answer_checks(GOLDEN_DATASET)[16]
    observed_answer = (
        "The Raval Shield has a DEF of 36, while the Raval Armor has a DEF of 48."
    )
    swapped_answer = (
        "The Raval Shield has a DEF of 48, while the Raval Armor has a DEF of 36."
    )

    observed = evaluate_answer(make_row(observed_answer), checks)
    reference = evaluate_answer(
        make_row("The Raval Shield provides 36 DEF, and the Raval Armor provides 48 DEF."),
        checks,
    )
    swapped = evaluate_answer(make_row(swapped_answer), checks)

    assert observed["answer_checks_status"] == "pass"
    assert observed["checks_passed"] == 2
    assert reference["answer_checks_status"] == "pass"
    assert swapped["answer_checks_status"] == "fail"
    assert swapped["checks_passed"] == 0
