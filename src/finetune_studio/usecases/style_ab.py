"""Blind A/B for the house-style template: which of two answers reads like the house style?

Per held-out question the judge gets the house-style reference answer (and the style guide, if any) and two answers
labelled 1 and 2, never which model wrote them. Every pair is asked twice with the order swapped, because LLM judges
favour a position (MT-Bench, arXiv 2306.05685); the trained model wins a question only when both orders pick it, the
base only when both pick the base, anything else is a tie. Wins vs losses (ties dropped) is a sign test, the same exact
binomial test as McNemar (``testing/paired.mcnemar_exact_p``). A failed or unreadable judge call is an error for that
question, never a verdict.
"""
from __future__ import annotations

import json
import re
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass, field
from typing import Any, Literal

from finetune_studio.testing.paired import NOISE_P, mcnemar_exact_p
from finetune_studio.usecases.metrics import strip_reasoning

AB_PROMPT_VERSION = "1"
PARALLEL_API = 4
Winner = Literal["1", "2", "tie", ""]
Outcome = Literal["tuned", "base", "tie", ""]
ChatFn = Callable[[list[dict[str, str]]], str]

AB_PROMPT = """You compare two answers for HOUSE STYLE only. You get a REQUEST, a REFERENCE answer written in the house style, optionally a STYLE GUIDE, and two candidate answers, ANSWER 1 and ANSWER 2.

Decide which candidate is closer to the house style of the reference: tone, formality, length, structure (lists, headings, paragraphs), how it opens and closes, wording habits. Do not reward an answer for being longer, more polite or more detailed unless the reference is. Ignore which answer has more facts, unless one says something the reference contradicts.

Reply with ONE JSON object and nothing else:
{"reasoning": "<one or two plain sentences>", "winner": "1" | "2" | "tie"}"""

_THINK = re.compile(r"<think>.*?</think>", re.DOTALL | re.IGNORECASE)
_WINNER = re.compile(r'"winner"\s*:\s*"?\s*(1|2|tie)\b', re.IGNORECASE)


@dataclass
class ABCase:
    key: str
    question: str
    reference: str
    base_answer: str
    tuned_answer: str


@dataclass
class ABResult:
    key: str
    outcome: Outcome
    first: Winner = ""        # order 1: ANSWER 1 = base
    second: Winner = ""       # order 2: ANSWER 1 = tuned
    reasoning: list[str] = field(default_factory=list)
    error: str = ""


def build_messages(case: ABCase, first: str, second: str, style_guide: str = "") -> list[dict[str, str]]:
    parts = [f"REQUEST:\n{case.question.strip()}", f"REFERENCE (house style):\n{case.reference.strip()}"]
    if style_guide.strip():
        parts.append(f"STYLE GUIDE:\n{style_guide.strip()}")
    parts += [f"ANSWER 1:\n{first.strip() or '(empty)'}", f"ANSWER 2:\n{second.strip() or '(empty)'}"]
    return [{"role": "system", "content": AB_PROMPT}, {"role": "user", "content": "\n\n".join(parts)}]


def parse_reply(raw: str) -> tuple[Winner, str]:
    """(winner, reasoning); winner '' when the reply names none."""
    text = _THINK.sub("", raw or "")
    for m in re.finditer(r"\{.*?\}", text, re.DOTALL):
        try:
            obj = json.loads(m.group(0))
        except ValueError:
            continue
        w = str(obj.get("winner", "")).strip().lower() if isinstance(obj, dict) else ""
        if w in ("1", "2", "tie"):
            return w, str(obj.get("reasoning") or "")  # type: ignore[return-value]
    m2 = _WINNER.search(text)
    return (m2.group(1).lower(), "") if m2 else ("", "")  # type: ignore[return-value]


def _ask(chat: ChatFn, messages: list[dict[str, str]]) -> tuple[Winner, str]:
    raw = chat(messages)
    winner, why = parse_reply(raw)
    if not winner:  # one reminder, like the main judge
        raw = chat([*messages, {"role": "assistant", "content": raw},
                    {"role": "user", "content": 'Reply with ONE JSON object only: {"reasoning": "...", "winner": "1|2|tie"}'}])
        winner, why = parse_reply(raw)
    return winner, why


def judge_case(chat: ChatFn, case: ABCase, style_guide: str = "") -> ABResult:
    """Both orders; the trained answer wins only when both orders pick it."""
    out = ABResult(key=case.key, outcome="")
    try:
        out.first, why1 = _ask(chat, build_messages(case, case.base_answer, case.tuned_answer, style_guide))
        if not out.first:
            out.error = "the judge reply named no winner"
            return out
        out.second, why2 = _ask(chat, build_messages(case, case.tuned_answer, case.base_answer, style_guide))
    except Exception as exc:  # noqa: BLE001 - provider errors vary; recorded, never a verdict
        out.error = f"judge call failed: {exc}"
        return out
    out.reasoning = [why1, why2]
    if not out.second:
        out.error = "the judge reply named no winner"
        return out
    tuned = (out.first == "2") + (out.second == "1")
    base = (out.first == "1") + (out.second == "2")
    out.outcome = "tuned" if tuned == 2 else "base" if base == 2 else "tie"
    return out


@dataclass(frozen=True)
class ABSummary:
    questions: int
    wins: int        # trained answer preferred in both orders
    losses: int      # base answer preferred in both orders
    ties: int        # a tie, or the two orders disagreed (position bias)
    errors: int
    win_rate: float | None   # wins / (wins + losses + ties)
    p_value: float
    verdict: str     # "trained preferred" | "base preferred" | "could be noise" | "nothing judged"

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


def summarize(results: list[ABResult]) -> ABSummary:
    wins = sum(r.outcome == "tuned" for r in results)
    losses = sum(r.outcome == "base" for r in results)
    ties = sum(r.outcome == "tie" for r in results)
    errors = sum(not r.outcome for r in results)
    judged = wins + losses + ties
    p = mcnemar_exact_p(wins, losses)
    if not judged:
        verdict = "nothing judged"
    elif p >= NOISE_P:
        verdict = "could be noise"
    else:
        verdict = "trained preferred" if wins > losses else "base preferred"
    return ABSummary(len(results), wins, losses, ties, errors, wins / judged if judged else None, p, verdict)


def run_ab(chat: ChatFn, cases: list[ABCase], *, style_guide: str = "", concurrent: bool = False,
           on_result: Callable[[ABResult], None] | None = None,
           should_stop: Callable[[], bool] = lambda: False) -> list[ABResult]:
    """Judge every case (a small thread pool for an API judge); ``on_result`` runs on the calling thread."""
    results: list[ABResult] = []

    def keep(r: ABResult) -> None:
        results.append(r)
        if on_result is not None:
            on_result(r)

    if not concurrent:
        for case in cases:
            if should_stop():
                break
            keep(judge_case(chat, case, style_guide))
        return results
    with ThreadPoolExecutor(max_workers=PARALLEL_API, thread_name_prefix="fts-ab") as pool:
        for start in range(0, len(cases), PARALLEL_API * 4):
            if should_stop():
                break
            batch = [pool.submit(judge_case, chat, c, style_guide) for c in cases[start:start + PARALLEL_API * 4]]
            for fut in batch:
                keep(fut.result())
    return results


def pair_cases(base_cases: list[dict[str, Any]], tuned_cases: list[dict[str, Any]]) -> list[ABCase]:
    """A/B cases for the questions both runs answered (matched by case name + question)."""
    def key(c: dict[str, Any]) -> str:
        return f"{c.get('case_name') or ''}\x1f{(c.get('question') or '').strip()}"

    def answered(c: dict[str, Any]) -> bool:
        return bool((c.get("model_answer") or "").strip())

    base = {key(c): c for c in base_cases if answered(c)}
    out = []
    for c in tuned_cases:
        k = key(c)
        if answered(c) and k in base:
            out.append(ABCase(key=k, question=c.get("question") or "", reference=c.get("correct_answer") or "",
                              base_answer=strip_reasoning(base[k].get("model_answer") or ""),
                              tuned_answer=strip_reasoning(c.get("model_answer") or "")))
    return out
