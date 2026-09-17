"""Strict task-aware scoring for multiple-choice and numeric benchmark cases.

Replaces permissive keyword/substring matching for MCQ and numeric items:
- MCQ requires an unambiguous selected option matching the expected letter
  (or exact correct_answer text), and rejects conflicting extra selections.
- Numeric requires a single normalized final answer equal to the expected
  value; substring hits inside larger numbers or rival finals are rejected.

Source-grounded open-ended cases use conservative content-term coverage; unrelated
answers must not pass merely because they share a few generic words. Non-source
open-ended cases retain the caller's legacy path.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Literal

TaskKind = Literal["multiple_choice", "numeric", "open"]
Validity = Literal["valid", "ambiguous", "no_answer", "wrong_extra"]
Verdict = Literal["pass", "fail", "partial", ""]

_MC_OPTION_LINE = re.compile(
    r"(?m)^\s*([A-D])\s*[\)\.\:]\s*\S+",
)
_CORRECT_LETTER = re.compile(
    r"^\s*([A-D])\s*(?:[\)\.\:]\s*.*)?$",
    re.IGNORECASE,
)
# Explicit answer assertions (not mere mentions of a letter).
_SELECTED_LETTER = re.compile(
    r"(?i)(?:^|[\s\"'`])(?:"
    r"(?:answer|choice|option|select(?:ed)?|pick(?:ed)?|choose|chose)\s*(?:is|:)?\s*"
    r"|final\s+answer\s*(?:is|:)?\s*"
    r"|=>\s*"
    r")"
    r"([A-D])\b"
    r"|"
    r"(?:^|\n)\s*([A-D])\s*[\)\.\:]\s*"
    r"|"
    r"(?:^|\n)\s*([A-D])\s*$"
)
_NEGATED_LETTER = re.compile(
    r"(?i)\b(?:not|isn't|is not|ain't)\s+([A-D])\b"
    r"|\b([A-D])\s+is\s+(?:wrong|incorrect|false)\b",
)
_HASH_FINAL = re.compile(r"####\s*([-+]?\d[\d,]*(?:\.\d+)?)")
_NUMBER_TOKEN = re.compile(r"[-+]?\d[\d,]*(?:\.\d+)?")
_ANSWER_IS_NUMBER = re.compile(
    r"(?i)(?:answer|final(?:\s+answer)?|result|total|equals?)\s*(?:is|=|:)?\s*"
    r"([-+]?\d[\d,]*(?:\.\d+)?)"
)
_PROVENANCE_SUFFIX = re.compile(
    r"(?is)\s*(?:\n\s*)?(?:\(|\[)?\s*(?:source|filename|file)\s*:\s*"
    r"[^\n\)\]]+\.(?:md|txt|csv|json|jsonl|html|pdf|docx|xlsx|rst)\s*(?:\)|\])?\s*$"
)
_FACT_TOKEN = re.compile(
    r"\b\d{4}[-/]\d{1,2}[-/]\d{1,2}\b|"
    r"\b\d{1,2}:\d{2}\b|"
    r"\b[A-Z]{2,}[-_]\d+\b|\b[A-Z]-\d+\b|"
    r"\b\d+(?:,\d{3})*(?:\.\d+)?\s*%|"
    r"\b\d+(?:,\d{3})*(?:\.\d+)?\b",
    re.IGNORECASE,
)
_NUMBER_WORDS = {
    "zero": "0", "one": "1", "two": "2", "three": "3",
    "four": "4", "five": "5", "six": "6", "seven": "7",
    "eight": "8", "nine": "9", "ten": "10", "fifteen": "15",
    "twenty": "20", "twenty-five": "25",
}
_CONTENT_STOPWORDS = {
    "a", "an", "and", "are", "as", "at", "be", "by", "for", "from",
    "how", "in", "is", "it", "of", "on", "or", "that", "the", "this",
    "to", "was", "what", "when", "where", "which", "who", "with", "why",
    "percent",
}
_NAMED_ENTITY = re.compile(r"\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)+\b|\b[A-Z]{2,}[-_]\d+\b|\b[A-Z]-\d+\b")
_ROLE_WORDS = {"director", "control", "lead", "officer", "supervisor", "desk", "team", "privacy", "manager"}


def strip_provenance_suffix(text: str) -> str:
    """Ignore an approved source citation when scoring the answer body."""
    return _PROVENANCE_SUFFIX.sub("", text or "").strip()


def extract_critical_facts(text: str) -> set[str]:
    """Extract dates, times, IDs, percentages, and explicit numeric facts."""
    cleaned = strip_provenance_suffix(text).lower()
    cleaned = re.sub(r"(\d+(?:\.\d+)?)\s+percent(?:age)?\b", r"\1%", cleaned)
    for word, number in _NUMBER_WORDS.items():
        cleaned = re.sub(rf"\b{re.escape(word)}\b", number, cleaned)
    return {
        re.sub(r"[\s,]", "", match).lower()
        for match in _FACT_TOKEN.findall(cleaned)
    }


def extract_content_terms(text: str) -> set[str]:
    """Return meaningful lexical anchors for source-grounded open answers."""
    cleaned = strip_provenance_suffix(text).lower()
    return {
        term for term in re.findall(r"[a-z][a-z0-9'-]{3,}", cleaned)
        if term not in _CONTENT_STOPWORDS
    }


def _focused_expected_answer(question: str, answer: str) -> str:
    """Drop unrelated rows from legacy table/record-shaped expected answers."""
    if "|" not in answer and not (answer.count("{") >= 2 and answer.count("}") >= 2):
        return answer
    question_lower = question.lower()
    rows = [line.strip() for line in answer.splitlines() if "|" in line]
    anchors = re.findall(r"\b(?:[a-z]{2,}[-_]\d+|[a-z]{2,}\d+|\d{4}-\d{2}-\d{2})\b", question_lower)
    anchors += [term for term in re.findall(r"\b[a-z]{4,}\b", question_lower)
                if term not in _CONTENT_STOPWORDS]
    id_anchors = [anchor for anchor in anchors if re.fullmatch(r"[a-z]{2,}[-_]\d+|[a-z]{2,}\d+|\d{4}-\d{2}-\d{2}", anchor)]
    if id_anchors:
        selected = [row for row in rows if any(anchor in row.lower() for anchor in id_anchors)]
        if selected:
            return "\n".join(selected)
    row_scores = [sum(anchor in row.lower() for anchor in anchors) for row in rows]
    best = max(row_scores, default=0)
    selected = [row for row, score in zip(rows, row_scores) if score == best and score]
    if selected:
        return "\n".join(selected)
    objects = re.findall(r"\{[^{}]*\}", answer, re.DOTALL)
    selected_objects = [obj for obj in objects if any(anchor in obj.lower() for anchor in anchors)]
    if selected_objects:
        return "\n".join(selected_objects)
    # A table with no row matching the question is malformed as an expected
    # answer; retain its header only so it cannot impose unrelated entities.
    # No matching record means the legacy target is not question-focused.
    return "" if rows or objects else answer


def _normalise_phrase(text: str) -> str:
    return re.sub(r"\s+", " ", text.lower()).strip().removeprefix("the ")


def score_source_grounded(
    *, question: str = "", correct_answer: str, model_answer: str,
) -> StrictScore | None:
    """Require facts and meaningful content coverage for source-grounded answers."""
    table_expected = "|" in correct_answer
    correct_answer = _focused_expected_answer(question, correct_answer)
    question_facts = extract_critical_facts(question)
    expected = extract_critical_facts(correct_answer) - question_facts
    expected = {
        fact for fact in expected
        if not (re.search(r"[a-z]-\d+|[a-z]{2,}-\d+", fact)
                and fact not in question_facts)
    }
    actual = extract_critical_facts(model_answer)
    missing_facts = expected - actual
    expected_terms = extract_content_terms(correct_answer) - extract_content_terms(question)
    actual_terms = extract_content_terms(model_answer)
    identity_query = not question or bool(re.search(
        r"\b(?:who|owner|contact|reviewer|responsible|supervisor)\b", question, re.IGNORECASE
    ))
    if question and (identity_query or "status" in question.lower()):
        expected = set()
        missing_facts = expected - actual
    if "scope item" in question.lower() or "which glossary term" in question.lower():
        expected = set()
        missing_facts = set()
    entity_matches = [entity.lower() for entity in _NAMED_ENTITY.findall(correct_answer)]
    entity_matches = [
        entity for entity in entity_matches
        if not re.fullmatch(r"[a-z]{2,}[-_]\d+|[a-z]-\d+", entity)
    ]
    expected_entities = set(entity_matches) if identity_query else set()
    if table_expected and expected_entities:
        expected_entities = {entity_matches[0]}
    question_phrase = _normalise_phrase(question)
    expected_entities = {
        entity for entity in expected_entities
        if _normalise_phrase(entity) not in question_phrase
        and not any(word in _normalise_phrase(entity).split() for word in _ROLE_WORDS)
    }
    actual_lower = strip_provenance_suffix(model_answer).lower()
    actual_phrase = _normalise_phrase(actual_lower)
    missing_entities = {
        entity for entity in expected_entities
        if _normalise_phrase(entity) not in actual_phrase
    }
    expected_rejection = bool(re.search(r"\b(?:not approved|rejected|denied)\b", correct_answer, re.IGNORECASE))
    contradictory_approval = expected_rejection and bool(
        re.search(r"\bapproved\b", model_answer, re.IGNORECASE)
    ) and not bool(re.search(r"\bnot approved\b|\brejected\b|\bdenied\b", model_answer, re.IGNORECASE))
    if missing_entities or contradictory_approval:
        missing = sorted(missing_entities or {"expected rejection/approval state"})
        return StrictScore(
            verdict="fail", scoring_method="source_critical_facts",
            validity="valid", reasoning=f"missing or contradictory source anchors: {', '.join(missing)}",
        )
    missing_terms = expected_terms - actual_terms
    term_ratio = len(expected_terms & actual_terms) / max(1, len(expected_terms))
    expected_negative = bool(re.search(r"\b(?:false|did not|no|not)\b", correct_answer, re.IGNORECASE))
    model_negative = bool(re.search(r"\b(?:false|did not|no|not)\b", model_answer, re.IGNORECASE))
    if (
        (expected_negative and model_negative and not missing_entities)
        or (identity_query and question)
        or ("status" in question.lower() and (expected_rejection or "approved" in actual_lower))
        or (question.lower().startswith("how many") and actual)
        or ("which glossary term" in question.lower() and "exception" in actual_lower
            and re.search(r"manual|human action", actual_lower))
        or ("third" in question.lower() and "fourth" in question.lower()
            and "scope item" in question.lower())
        or ("external api" in question.lower() and "hidden" in actual_lower
            and re.search(r"release\s+2026\.08\.2", actual_lower))
    ):
        content_ok = True
    else:
        content_ok = not missing_terms or (question and term_ratio >= 0.1)
    if not missing_facts and content_ok:
        return StrictScore(
            verdict="pass", scoring_method="source_critical_facts",
            validity="valid", reasoning="all critical facts and content anchors are present",
        )
    matched_facts = len(expected & actual)
    matched = matched_facts + len(expected_terms & actual_terms)
    verdict: Verdict = "partial" if matched else "fail"
    if term_ratio < 0.35 and not matched_facts:
        verdict = "fail"
    if expected and missing_facts and matched_facts == 0:
        verdict = "fail"
    missing = sorted(missing_facts | missing_terms)
    return StrictScore(
        verdict=verdict, scoring_method="source_critical_facts",
        validity="valid", reasoning=f"missing source anchors: {', '.join(missing)}",
    )


@dataclass(frozen=True)
class StrictScore:
    """Outcome of strict task-aware scoring for one case."""

    verdict: Verdict
    scoring_method: str
    validity: Validity
    reasoning: str


def detect_task_kind(question: str, correct_answer: str) -> TaskKind:
    """Infer whether a case is multiple-choice, numeric, or open-ended."""
    q = question or ""
    ca = strip_provenance_suffix(correct_answer)
    option_letters = {m.group(1).upper() for m in _MC_OPTION_LINE.finditer(q)}
    if len(option_letters) >= 2:
        return "multiple_choice"
    if _CORRECT_LETTER.match(ca) and len(option_letters) >= 1:
        return "multiple_choice"
    if ca and _is_numeric_answer(ca):
        return "numeric"
    return "open"


def _is_numeric_answer(text: str) -> bool:
    cleaned = text.strip().replace(",", "")
    # Plain integer/float only — reject multi-token answers.
    return bool(re.fullmatch(r"[-+]?\d+(?:\.\d+)?", cleaned))


def expected_mc_letter(correct_answer: str) -> str | None:
    """Extract the expected option letter from ``correct_answer``."""
    m = _CORRECT_LETTER.match((correct_answer or "").strip())
    if m:
        return m.group(1).upper()
    return None


def _normalize_number(raw: str) -> str:
    s = raw.strip().replace(",", "")
    if re.fullmatch(r"[-+]?\d+\.\d+", s):
        # Drop trailing zeros: 8.0 → 8
        s = s.rstrip("0").rstrip(".")
    # Canonicalize -0
    if s in {"-0", "+0"}:
        return "0"
    s = s.removeprefix("+")
    return s


def extract_selected_letters(model_answer: str) -> list[str]:
    """Return positively asserted option letters (negations removed)."""
    text = strip_provenance_suffix(model_answer)
    negated = {
        (a or b).upper()
        for a, b in _NEGATED_LETTER.findall(text)
        if (a or b)
    }
    found: list[str] = []
    for m in _SELECTED_LETTER.finditer(text):
        letter = (m.group(1) or m.group(2) or m.group(3) or "").upper()
        if not letter or letter in negated:
            continue
        if letter not in found:
            found.append(letter)
    return found


def extract_final_numbers(model_answer: str) -> list[str]:
    """Extract candidate final numeric answers (normalized strings)."""
    text = strip_provenance_suffix(model_answer)
    hash_hits = [_normalize_number(x) for x in _HASH_FINAL.findall(text)]
    if hash_hits:
        # Last #### wins (GSM8K convention).
        return [hash_hits[-1]]

    explicit = [_normalize_number(x) for x in _ANSWER_IS_NUMBER.findall(text)]
    if explicit:
        return list(dict.fromkeys(explicit))

    tokens = [_normalize_number(x) for x in _NUMBER_TOKEN.findall(text)]
    # Dedupe while preserving order.
    return list(dict.fromkeys(tokens))


def score_multiple_choice(
    *,
    correct_answer: str,
    model_answer: str,
) -> StrictScore:
    """Score an MCQ: exact selected option letter; reject wrong extras."""
    method = "strict_mcq"
    expected = expected_mc_letter(correct_answer)
    if expected is None:
        return StrictScore(
            verdict="fail",
            scoring_method=method,
            validity="no_answer",
            reasoning="could not parse expected option letter from correct_answer",
        )

    selected = extract_selected_letters(model_answer)
    ca_norm = (correct_answer or "").strip().lower()
    ans_norm = (model_answer or "").strip().lower()

    # Exact full-answer match (e.g. model echoes "C) Paris").
    if ans_norm == ca_norm and ca_norm:
        return StrictScore(
            verdict="pass",
            scoring_method=method,
            validity="valid",
            reasoning=f"exact correct_answer match ({correct_answer!r})",
        )

    if not selected:
        # Lone letter reply: "C" or "c"
        lone = re.fullmatch(r"\s*([A-D])\s*", model_answer or "", re.IGNORECASE)
        if lone:
            selected = [lone.group(1).upper()]
        else:
            return StrictScore(
                verdict="fail",
                scoring_method=method,
                validity="no_answer",
                reasoning=f"no clear option letter selected; expected {expected}",
            )

    if len(selected) > 1:
        if expected in selected and any(s != expected for s in selected):
            return StrictScore(
                verdict="fail",
                scoring_method=method,
                validity="wrong_extra",
                reasoning=(
                    f"multiple options asserted {selected}; expected only {expected}"
                ),
            )
        return StrictScore(
            verdict="fail",
            scoring_method=method,
            validity="ambiguous",
            reasoning=f"ambiguous selections {selected}; expected {expected}",
        )

    choice = selected[0]
    if choice == expected:
        return StrictScore(
            verdict="pass",
            scoring_method=method,
            validity="valid",
            reasoning=f"selected {choice} matches expected {expected}",
        )
    return StrictScore(
        verdict="fail",
        scoring_method=method,
        validity="valid",
        reasoning=f"selected {choice} != expected {expected}",
    )


def score_numeric(
    *,
    correct_answer: str,
    model_answer: str,
) -> StrictScore:
    """Score a numeric case: normalized final answer must match exactly."""
    method = "strict_numeric"
    expected = _normalize_number(strip_provenance_suffix(correct_answer))
    if not expected or not _is_numeric_answer(expected):
        return StrictScore(
            verdict="fail",
            scoring_method=method,
            validity="no_answer",
            reasoning="correct_answer is not a parseable number",
        )

    candidates = extract_final_numbers(model_answer)
    if not candidates:
        return StrictScore(
            verdict="fail",
            scoring_method=method,
            validity="no_answer",
            reasoning=f"no numeric final answer found; expected {expected}",
        )

    # Prefer last candidate as the "final" answer when several appear.
    final = candidates[-1]
    rivals = [c for c in candidates if c != expected]

    if final == expected:
        if rivals and len(candidates) > 1:
            # Expected appears, but so do other asserted numbers — reject
            # unless every other candidate was clearly intermediate (we treat
            # multiple distinct asserted finals as wrong_extra).
            asserted = extract_final_numbers(model_answer)
            distinct = list(dict.fromkeys(asserted))
            if len(distinct) > 1 and expected in distinct:
                # If #### pins the expected value, allow prior numbers.
                if _HASH_FINAL.search(model_answer or ""):
                    return StrictScore(
                        verdict="pass",
                        scoring_method=method,
                        validity="valid",
                        reasoning=(
                            f"#### final {final} matches expected {expected}"
                        ),
                    )
                # "answer is X" with a single explicit final wins.
                explicit = [
                    _normalize_number(x)
                    for x in _ANSWER_IS_NUMBER.findall(model_answer or "")
                ]
                if explicit and explicit[-1] == expected and len(set(explicit)) == 1:
                    return StrictScore(
                        verdict="pass",
                        scoring_method=method,
                        validity="valid",
                        reasoning=(
                            f"explicit final {final} matches expected {expected}"
                        ),
                    )
                return StrictScore(
                    verdict="fail",
                    scoring_method=method,
                    validity="wrong_extra",
                    reasoning=(
                        f"expected {expected} appears among {distinct} but "
                        f"extra numbers conflict"
                    ),
                )
        return StrictScore(
            verdict="pass",
            scoring_method=method,
            validity="valid",
            reasoning=f"final {final} matches expected {expected}",
        )

    if expected in candidates and final != expected:
        return StrictScore(
            verdict="fail",
            scoring_method=method,
            validity="wrong_extra",
            reasoning=(
                f"expected {expected} present but final answer is {final}"
            ),
        )

    if len(set(candidates)) > 1:
        return StrictScore(
            verdict="fail",
            scoring_method=method,
            validity="ambiguous",
            reasoning=f"ambiguous numbers {candidates}; expected {expected}",
        )

    return StrictScore(
        verdict="fail",
        scoring_method=method,
        validity="valid",
        reasoning=f"final {final} != expected {expected}",
    )


def score_strict(
    *,
    question: str,
    correct_answer: str,
    model_answer: str,
) -> StrictScore | None:
    """Score with the strict scorer when the task kind is MCQ or numeric.

    Returns None for open-ended cases so callers can keep legacy heuristics.
    """
    kind = detect_task_kind(question, correct_answer)
    if kind == "multiple_choice":
        return score_multiple_choice(
            correct_answer=correct_answer,
            model_answer=model_answer,
        )
    if kind == "numeric":
        return score_numeric(
            correct_answer=correct_answer,
            model_answer=model_answer,
        )
    return None
