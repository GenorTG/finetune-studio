"""IFEval (google/IFEval): verifiable instruction following, strict prompt-level scoring.

Fact or style fine-tuning can quietly cost chat ability: in Thinking Machines' company-docs run IFEval fell
from 85 to 45 while internal-QA recall rose (On-Policy Distillation, 2025-10-27). MMLU/GSM8K do not see
that, so this suite runs the base model and a tuned run on the same prompts and the Compare tab pairs them.

The checkers re-implement google-research ``instruction_following_eval/instructions.py`` (strict mode) for
every instruction type that needs no extra library. Prompts that use ``language:response_language``
(needs langdetect) or ``length_constraints:number_sentences`` (needs the official sentence splitter) are
left out instead of approximated, so the score is "strict accuracy over the supported prompts", close to but
not the official number. A prompt passes only when every one of its instructions passes.
"""

from __future__ import annotations

import json
import re
from collections.abc import Callable, Mapping
from typing import Any

ANSWER_PREFIX = "IFEVAL:"
SCORING_METHOD = "ifeval_strict"

_WORD = re.compile(r"\w+")


def _compare(count: int, relation: str | None, n: int) -> bool:
    if relation == "less than":
        return count < n
    if relation == "at least":
        return count >= n
    raise ValueError(f"unknown relation {relation!r}")


def _no_comma(r: str, _: Mapping[str, Any]) -> bool:
    return "," not in r


def _number_words(r: str, k: Mapping[str, Any]) -> bool:
    return _compare(len(_WORD.findall(r)), k["relation"], int(k["num_words"]))


def _forbidden_words(r: str, k: Mapping[str, Any]) -> bool:
    return not any(re.search(r"\b" + re.escape(w) + r"\b", r, re.IGNORECASE) for w in k["forbidden_words"] or [])


def _highlighted_sections(r: str, k: Mapping[str, Any]) -> bool:
    found = [h for h in re.findall(r"\*[^\n\*]*\*", r) if h.strip("*").strip()]
    found += [h for h in re.findall(r"\*\*[^\n\*]*\*\*", r) if h.removeprefix("**").removesuffix("**").strip()]
    return len(found) >= int(k["num_highlights"])


def _keyword_frequency(r: str, k: Mapping[str, Any]) -> bool:
    return _compare(len(re.findall(re.escape(k["keyword"]), r, re.IGNORECASE)), k["relation"], int(k["frequency"]))


def _repeat_prompt(r: str, k: Mapping[str, Any]) -> bool:
    return r.strip().lower().startswith(str(k["prompt_to_repeat"]).strip().lower())


def _quotation(r: str, _: Mapping[str, Any]) -> bool:
    v = r.strip()
    return len(v) > 1 and v[0] == '"' and v[-1] == '"'


def _lowercase(r: str, _: Mapping[str, Any]) -> bool:
    return r.islower()


def _capital(r: str, _: Mapping[str, Any]) -> bool:
    return r.isupper()


def _keywords_exist(r: str, k: Mapping[str, Any]) -> bool:
    return all(re.search(re.escape(w), r, re.IGNORECASE) for w in k["keywords"] or [])


def _title(r: str, _: Mapping[str, Any]) -> bool:
    return any(t.lstrip("<").rstrip(">").strip() for t in re.findall(r"<<[^\n]+>>", r))


def _letter_frequency(r: str, k: Mapping[str, Any]) -> bool:
    return _compare(r.lower().count(str(k["letter"]).lower()), k["let_relation"], int(k["let_frequency"]))


def _bullet_lists(r: str, k: Mapping[str, Any]) -> bool:
    stars = re.findall(r"^\s*\*[^\*].*$", r, re.MULTILINE)
    dashes = re.findall(r"^\s*-.*$", r, re.MULTILINE)
    return len(stars) + len(dashes) == int(k["num_bullets"])


def _placeholders(r: str, k: Mapping[str, Any]) -> bool:
    return len(re.findall(r"\[.*?\]", r)) >= int(k["num_placeholders"])


def _paragraphs(r: str, k: Mapping[str, Any]) -> bool:
    parts = re.split(r"\s?\*\*\*\s?", r)
    count = len(parts)
    for i, part in enumerate(parts):
        if not part.strip():
            if i in (0, len(parts) - 1):
                count -= 1
            else:
                return False
    return count == int(k["num_paragraphs"])


def _end_checker(r: str, k: Mapping[str, Any]) -> bool:
    return r.strip().strip('"').lower().endswith(str(k["end_phrase"]).strip().lower())


def _postscript(r: str, k: Mapping[str, Any]) -> bool:
    marker = str(k["postscript_marker"]).lower()
    if marker == "p.p.s":
        pattern = r"\s*p\.\s?p\.\s?s.*$"
    elif marker == "p.s.":
        pattern = r"\s*p\.\s?s\..*$"
    else:
        pattern = r"\s*" + re.escape(marker) + r".*$"
    return bool(re.findall(pattern, r.lower(), re.MULTILINE))


def _capital_word_frequency(r: str, k: Mapping[str, Any]) -> bool:
    caps = [w for w in _WORD.findall(r) if w.isupper()]
    return _compare(len(caps), k["capital_relation"], int(k["capital_frequency"]))


def _two_responses(r: str, _: Mapping[str, Any]) -> bool:
    parts = r.split("******")
    valid = []
    for i, part in enumerate(parts):
        if not part.strip():
            if i not in (0, len(parts) - 1):
                return False
        else:
            valid.append(part)
    return len(valid) == 2 and valid[0].strip() != valid[1].strip()


def _json_format(r: str, _: Mapping[str, Any]) -> bool:
    v = r.strip()
    for prefix in ("```json", "```Json", "```JSON", "```"):
        v = v.removeprefix(prefix)
    v = v.removesuffix("```").strip()
    try:
        json.loads(v)
    except ValueError:
        return False
    return True


def _multiple_sections(r: str, k: Mapping[str, Any]) -> bool:
    pattern = r"\s?" + re.escape(str(k["section_spliter"]).strip()) + r"\s?\d+\s?"
    return len(re.split(pattern, r)) - 1 >= int(k["num_sections"])


def _nth_paragraph_first_word(r: str, k: Mapping[str, Any]) -> bool:
    parts = re.split(r"\n\n", r)
    count = len(parts) - sum(1 for p in parts if not p.strip())
    nth = int(k["nth_paragraph"])
    if nth > count:
        return False
    para = parts[nth - 1].strip()
    if not para:
        return False
    word = para.split()[0].strip().lstrip("'").lstrip('"')
    first = ""
    for ch in word:
        if ch in {".", ",", "?", "!", "'", '"'}:
            break
        first += ch.lower()
    return count == int(k["num_paragraphs"]) and first == str(k["first_word"]).lower()


def _constrained_response(r: str, _: Mapping[str, Any]) -> bool:
    return any(opt in r.strip() for opt in ("My answer is yes.", "My answer is no.", "My answer is maybe."))


CHECKERS: dict[str, Callable[[str, Mapping[str, Any]], bool]] = {
    "punctuation:no_comma": _no_comma,
    "length_constraints:number_words": _number_words,
    "keywords:forbidden_words": _forbidden_words,
    "detectable_format:number_highlighted_sections": _highlighted_sections,
    "keywords:frequency": _keyword_frequency,
    "combination:repeat_prompt": _repeat_prompt,
    "startend:quotation": _quotation,
    "change_case:english_lowercase": _lowercase,
    "keywords:existence": _keywords_exist,
    "detectable_format:title": _title,
    "keywords:letter_frequency": _letter_frequency,
    "detectable_format:number_bullet_lists": _bullet_lists,
    "detectable_content:number_placeholders": _placeholders,
    "length_constraints:number_paragraphs": _paragraphs,
    "startend:end_checker": _end_checker,
    "detectable_content:postscript": _postscript,
    "change_case:english_capital": _capital,
    "change_case:capital_word_frequency": _capital_word_frequency,
    "combination:two_responses": _two_responses,
    "detectable_format:json_format": _json_format,
    "detectable_format:multiple_sections": _multiple_sections,
    "length_constraints:nth_paragraph_first_word": _nth_paragraph_first_word,
    "detectable_format:constrained_response": _constrained_response,
}


def is_supported(instruction_ids: list[str]) -> bool:
    """True when every instruction of a prompt has a checker here."""
    return bool(instruction_ids) and all(i in CHECKERS for i in instruction_ids)


def encode_answer(instruction_ids: list[str], kwargs: list[Mapping[str, Any]]) -> str:
    """The answer key stored on the case: the instructions to verify, not a reference text."""
    clean = [{k: v for k, v in (kw or {}).items() if v is not None} for kw in kwargs]
    return ANSWER_PREFIX + json.dumps({"ids": list(instruction_ids), "kwargs": clean}, ensure_ascii=False)


def is_ifeval_answer(correct_answer: str) -> bool:
    return (correct_answer or "").startswith(ANSWER_PREFIX)


def check_response(correct_answer: str, response: str) -> tuple[bool, list[str]]:
    """(every instruction followed, ids that failed) for one prompt's encoded answer key."""
    spec = json.loads(correct_answer[len(ANSWER_PREFIX):])
    failed = []
    for iid, kw in zip(spec["ids"], spec["kwargs"], strict=True):
        ok = bool(response.strip()) and CHECKERS[iid](response, kw)
        if not ok:
            failed.append(iid)
    return not failed, failed
