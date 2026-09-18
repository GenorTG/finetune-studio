"""Regression tests: strict MCQ/numeric scoring rejects keyword false positives."""

from __future__ import annotations

from finetune_studio.testing.strict_scoring import (
    detect_task_kind,
    score_multiple_choice,
    score_numeric,
    score_source_grounded,
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


def test_source_grounded_open_answer_rejects_unrelated_proper_action() -> None:
    s = score_source_grounded(
        correct_answer="Management must close the maintenance action by 2026-08-15.",
        model_answer="The next maintenance action is due no later than 2026-08-15.",
    )
    assert s is not None
    assert s.verdict in {"fail", "partial"}


def test_source_grounded_open_answer_rejects_wrong_named_contact() -> None:
    s = score_source_grounded(
        correct_answer="Pavel Novak is the Rotterdam maintenance contact.",
        model_answer="Mira Varga is the Rotterdam maintenance contact.",
    )
    assert s is not None
    assert s.verdict == "fail"


def test_source_grounded_rejects_contradictory_approval_state() -> None:
    s = score_source_grounded(
        correct_answer="The request is not approved. Operations rejected it because carrier rate limits are unknown.",
        model_answer="CR-77 is approved pending a carrier scan.",
    )
    assert s is not None
    assert s.verdict == "fail"


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


def test_source_grounded_scoring_rejects_wrong_date() -> None:
    score = score_source_grounded(
        correct_answer="The review is scheduled for 2026-10-05.",
        model_answer="The review is scheduled for 2026-11-15.",
    )
    assert score is not None
    assert score.verdict == "fail"


def test_source_grounded_scoring_accepts_number_words() -> None:
    score = score_source_grounded(
        correct_answer="Two of 86 lots exceeded the 2 percent rule.",
        model_answer="2 of 86 lots exceeded the 2% rule.",
    )
    assert score is not None
    assert score.verdict == "pass"


def test_source_grounded_scoring_accepts_plural_zero_word() -> None:
    score = score_source_grounded(
        question="What changed for SKU values?",
        correct_answer="A leading zero is now preserved in SKU values.",
        model_answer="Leading zeros are now preserved in SKU values.",
    )
    assert score is not None
    assert score.verdict == "pass"


def test_source_grounded_does_not_require_unasked_date_or_site() -> None:
    lane = score_source_grounded(
        question="Which lane was activated when C-17 stopped?",
        correct_answer="Lane C-12 was activated on 2026-08-04 at Rotterdam.",
        model_answer="Lane C-12 was activated.",
    )
    case = score_source_grounded(
        question="What is the case identifier and what did the customer report?",
        correct_answer="Case CS-491 reported a missing parcel on 2026-08-09.",
        model_answer="The case is CS-491 and the customer reported a missing parcel.",
    )
    assert lane is not None and lane.verdict == "pass"
    assert case is not None and case.verdict == "pass"


def test_source_grounded_still_requires_date_when_question_asks_when() -> None:
    score = score_source_grounded(
        question="When was C-17 restored?",
        correct_answer="C-17 was restored on 2026-08-04 at 10:18.",
        model_answer="C-17 was restored at 10:18.",
    )
    assert score is not None
    assert score.verdict == "partial"


def test_source_grounded_purpose_answer_can_be_concise() -> None:
    precise = score_source_grounded(
        question="What is the purpose of including the scanner identifier?",
        correct_answer=(
            "The scanner identifier is one required field and provides traceability "
            "of which scanner produced the failed read."
        ),
        model_answer="It helps trace which scanner produced the exception.",
    )
    vague = score_source_grounded(
        question="What is the overall purpose of the Customer Communication Guide?",
        correct_answer=(
            "It provides a standard for known facts, checks, update timing, refunds, "
            "card numbers, and escalation to the duty manager."
        ),
        model_answer="The guide provides a standard for communicating with customers about delays.",
    )
    assert precise is not None and precise.verdict == "pass"
    assert vague is not None and vague.verdict == "partial"


def test_source_grounded_ignores_unrelated_rows_in_legacy_table_answer() -> None:
    score = score_source_grounded(
        question="Who is the recorded owner of risk RK-04?",
        correct_answer=(
            "risk_id | owner\n--- | ---\n"
            "RK-01 | Pavel Novak\nRK-04 | Elian Mertens"
        ),
        model_answer="Elian Mertens is the owner of risk RK-04.",
    )
    assert score is not None
    assert score.verdict == "pass"


def test_source_grounded_table_focus_still_rejects_wrong_owner() -> None:
    score = score_source_grounded(
        question="Who is the recorded owner of risk RK-04?",
        correct_answer="risk_id | owner\n--- | ---\nRK-04 | Elian Mertens",
        model_answer="Nadiya Petrov is the owner of risk RK-04.",
    )
    assert score is not None
    assert score.verdict == "fail"


def test_question_supplied_identifier_is_not_required_in_answer() -> None:
    score = score_source_grounded(
        question="What is the approval status of CR-77?",
        correct_answer="CR-77 is not approved.",
        model_answer="The request is not approved.",
    )
    assert score is not None
    assert score.verdict == "pass"


def test_question_aware_scoring_still_rejects_wrong_date() -> None:
    score = score_source_grounded(
        question="When is the review scheduled?",
        correct_answer="The review is scheduled for 2026-10-05.",
        model_answer="The review is scheduled for 2026-11-15.",
    )
    assert score is not None
    assert score.verdict == "fail"


def test_external_api_answer_is_not_penalized_by_unrelated_metrics_table() -> None:
    score = score_source_grounded(
        question="What is the external API status for project OCTOPUS-7741, and which release will revert it?",
        correct_answer=(
            "metric | value | project | date\n--- | --- | --- | ---\n"
            "external_api_hidden | true | OCTOPUS-7741 | 2026-09-09"
        ),
        model_answer=(
            "The external API fields remain hidden; the change is reverted in release 2026.08.2."
        ),
    )
    assert score is not None
    assert score.verdict == "pass"


def test_question_focused_scope_and_glossary_answers_pass_without_extra_prose() -> None:
    scope = score_source_grounded(
        question="What are the third and fourth scope items listed in the brief?",
        correct_answer="The third scope item is Tool-calling chat, and the fourth is an E2E WebUI stress test dated 2026-09-10.",
        model_answer="The third scope item is Tool-calling chat, and the fourth is an E2E WebUI stress test.",
    )
    glossary = score_source_grounded(
        question="Which glossary term specifically requires human action?",
        correct_answer="Exception requires human action after the third failure.",
        model_answer="The glossary term is exception, which requires human action.",
    )
    assert scope is not None and scope.verdict == "pass"
    assert glossary is not None and glossary.verdict == "pass"
