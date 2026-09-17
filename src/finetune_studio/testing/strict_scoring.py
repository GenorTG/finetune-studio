"""Strict task-aware scoring for multiple-choice and numeric benchmark cases.

Replaces permissive keyword/substring matching for MCQ and numeric items:
- MCQ requires an unambiguous selected option matching the expected letter
  (or exact correct_answer text), and rejects conflicting extra selections.
- Numeric requires a single normalized final answer equal to the expected
  value; substring hits inside larger numbers or rival finals are rejected.

Open-ended cases fall back to the caller (legacy keyword / heuristic path).
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


def strip_provenance_suffix(text: str) -> str:
    """Ignore an approved source citation when scoring the answer body."""
    return _PROVENANCE_SUFFIX.sub("", text or "").strip()


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
