"""Regression tests: strict MCQ/numeric scoring rejects keyword false positives."""

from __future__ import annotations

from finetune_studio.testing.strict_scoring import (
    detect_task_kind,
    score_multiple_choice,
    score_numeric,
    score_strict,
)
from finetune_studio.testing.suite import CaseResult, apply_heuristic_judging

_MCQ_Q = (
    "What is the capital of France?\n"
    "A) Berlin\nB) Madrid\nC) Paris\nD) Rome\n"
    "Reply with the letter and the answer."
)


def test_detect_mcq_and_numeric() -> None:
    assert detect_task_kind(_MCQ_Q, "C) Paris") == "multiple_choice"
    assert detect_task_kind("How many left?", "8") == "numeric"
    assert detect_task_kind("Explain LoRA.", "adapters") == "open"


def test_mcq_pass_on_exact_letter() -> None:
    s = score_multiple_choice(
        correct_answer="C) Paris",
        model_answer="Answer: C) Paris",
    )
    assert s.verdict == "pass"
    assert s.scoring_method == "strict_mcq"
    assert s.validity == "valid"


def test_mcq_rejects_keyword_false_positive_mentions_correct_text() -> None:
    """Legacy keywords ['Paris','C'] would pass; strict must fail wrong letter."""
    s = score_multiple_choice(
        correct_answer="C) Paris",
        model_answer="A) Berlin — Paris is in France but I pick Berlin.",
    )
    assert s.verdict == "fail"
    assert s.scoring_method == "strict_mcq"


def test_mcq_rejects_wrong_extra_when_both_letters_asserted() -> None:
    s = score_multiple_choice(
        correct_answer="C) Paris",
        model_answer="The answer is A or C.",
    )
    # "answer is A" + trailing "C" may parse as multiple — must not pass.
    assert s.verdict == "fail"
    assert s.validity in {"wrong_extra", "ambiguous", "valid"}


def test_mcq_rejects_substring_letter_without_selection() -> None:
    s = score_multiple_choice(
        correct_answer="C) Paris",
        model_answer="I am Completely unsure about this question.",
    )
    assert s.verdict == "fail"
    assert s.validity == "no_answer"


def test_numeric_pass_on_normalized_final() -> None:
    s = score_numeric(correct_answer="8", model_answer="#### 8")
    assert s.verdict == "pass"
    assert s.scoring_method == "strict_numeric"
    assert s.validity == "valid"


def test_numeric_ignores_approved_filename_citation() -> None:
    s = score_numeric(
        correct_answer="8",
        model_answer="The total is 8. (Source: 15_overtime_budget.csv)",
    )
    assert s.verdict == "pass"


def test_training_target_strips_citation_but_keeps_answer() -> None:
    from finetune_studio.training.data import format_for_sft

    out = format_for_sft([{
        "conversations": [
            {"from": "human", "value": "What is the total?"},
            {"from": "gpt", "value": "8 (Source: 15_overtime_budget.csv)"},
        ],
    }])
    assert out[0]["messages"][1]["content"] == "8"


def test_numeric_rejects_substring_inside_larger_number() -> None:
    """Keyword '8' would match inside '18' / '48'; strict must fail."""
    s = score_numeric(correct_answer="8", model_answer="The total is 18.")
    assert s.verdict == "fail"
    assert s.scoring_method == "strict_numeric"


def test_numeric_rejects_wrong_extra_conflicting_finals() -> None:
    s = score_numeric(
        correct_answer="8",
        model_answer="I get 7 and also 8 as possible answers.",
    )
    assert s.verdict == "fail"
    assert s.validity in {"wrong_extra", "ambiguous"}


def test_apply_heuristic_records_scoring_metadata() -> None:
    results = [
        CaseResult(
            case_name="mcq",
            category="knowledge",
            question=_MCQ_Q,
            correct_answer="C) Paris",
            model_answer="C",
            keywords=["Paris", "C"],
        ),
        CaseResult(
            case_name="fp",
            category="knowledge",
            question=_MCQ_Q,
            correct_answer="C) Paris",
            model_answer="A) Berlin mentions Paris somewhere.",
            keywords=["Paris", "C"],
        ),
        CaseResult(
            case_name="num_fp",
            category="math",
            question="How many apples are left? Give only the final number.",
            correct_answer="8",
            model_answer="48",
            keywords=["8"],
        ),
    ]
    apply_heuristic_judging(results)
    assert results[0].verdict == "pass"
    assert results[0].scoring_method == "strict_mcq"
    assert results[0].validity == "valid"
    assert "strict_mcq" in results[0].judge_reasoning

    assert results[1].verdict == "fail"
    assert results[1].scoring_method == "strict_mcq"

    assert results[2].verdict == "fail"
    assert results[2].scoring_method == "strict_numeric"


def test_score_strict_returns_none_for_open() -> None:
    assert score_strict(
        question="What is LoRA?",
        correct_answer="Low-Rank Adaptation",
        model_answer="LoRA adapters",
    ) is None
