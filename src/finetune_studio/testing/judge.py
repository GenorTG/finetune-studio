"""The AI judge: decides afterwards whether a saved model answer holds the information the answer key holds.

A test run only records what the model was asked, what it answered and what the correct answer is. Nothing in
this module (or anywhere on the custom-dataset test path) compares strings: a model may word, order or format a
correct answer in a way no matcher anticipates, so a *model* reads each saved case the way a person would and
returns pass / partial / fail with its reasoning. Any provider row can be the judge (local helper GGUF, API
helper, any OpenAI-compatible provider); a human can override any verdict in the UI.

Failure is never converted into a verdict: when the judge call fails or its reply cannot be parsed the case
stays unjudged and the error is recorded, so a flaky judge cannot silently fail (or pass) a model.
"""

from __future__ import annotations

import json
import logging
import re
from collections.abc import Callable, Iterator
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Any, Literal

log = logging.getLogger(__name__)

Verdict = Literal["pass", "fail", "partial", ""]
VERDICTS = ("pass", "partial", "fail")
JUDGE_PROMPT_VERSION = "4"
JUDGE_MAX_TOKENS = 800
JUDGE_PARALLEL_API = 4  # concurrent requests to an API provider; a local GGUF is one GPU and runs one at a time

ChatFn = Callable[[list[dict]], str]

JUDGE_PROMPT = """You are a strict but fair grader. A model under test was asked a question; the ANSWER KEY holds the information a correct answer must contain. Decide whether the MODEL ANSWER holds that information.

You get: QUESTION, ANSWER KEY (the reference answer from the source documents), optionally KEY VALUES (values a complete answer must contain), and MODEL ANSWER.

Grade the information, not the wording. Ignore style, tone, length, structure and ordering: a 5-word answer and a 500-word answer can both be right.

Procedure (for a normal case):
1. List every distinct fact the answer key requires: one entry per value, name, number, date or relationship.
2. For each fact give a status and copy the EXACT words of the model answer that state it into "evidence" (empty when there are none):
   - "present": the model answer states the same value. Only a pure change of form is the same value: 44.20 = 44.2, "four" = 4, 12,750 = 12750, "EUR 14.5k" = 14,500 euros, 4 March 2026 = 2026-03-04, a list in another order.
   - "wrong": the model answer gives a different value for it (another number, name, date, unit) or contradicts it.
   - "missing": the model answer does not say it, or only names the topic.
   Never infer one value from another. A week number is not a date, a role is not a name, a nearby or enclosing value is not the value, a range is not the value inside it, a similar-looking identifier is not the identifier.
3. Set "extra_wrong" to true only when the model answer adds a claim that contradicts the key or invents a conflicting specific. Extra correct detail is fine.
4. Verdict: "pass" = every fact present and extra_wrong false; "fail" = no fact present (wrong, missing, irrelevant, or a refusal / "I don't know" to a question the key answers: the case type says whether the documents hold the answer, so never speculate that they might not); "partial" = anything in between.

If the case is marked UNANSWERABLE, the documents do NOT contain the answer, so the only correct behaviour is to say the information is not available / not in the documents. Use an empty "facts" list. "pass" = the model clearly declines and invents nothing; "partial" = it hedges but still offers made-up specifics; "fail" = it answers confidently with specifics.

Reply with ONE JSON object and nothing else:
{"facts": [{"fact": "<required fact>", "status": "present" | "wrong" | "missing", "evidence": "<exact words from the model answer, or empty>"}], "extra_wrong": true | false, "reasoning": "<one to three plain sentences>", "verdict": "pass" | "partial" | "fail", "confidence": <0.0 to 1.0>}"""


@dataclass
class JudgeCase:
    """Everything the judge sees about one saved case."""

    question: str
    correct_answer: str
    model_answer: str
    keywords: list[str] = field(default_factory=list)
    expect_abstain: bool = False


@dataclass
class JudgeResult:
    """One judge call. ``verdict`` is empty when the call failed — ``error`` then says why."""

    verdict: Verdict = ""
    reasoning: str = ""
    confidence: float | None = None
    error: str = ""
    raw: str = ""
    facts: list[dict[str, Any]] = field(default_factory=list)  # the judge's per-fact checklist (rubric), when it gave one


def build_judge_messages(case: JudgeCase) -> list[dict]:
    """Chat messages for one case."""
    parts = [f"QUESTION:\n{case.question.strip()}"]
    if case.expect_abstain:
        parts.append(
            "CASE TYPE: UNANSWERABLE — the documents do not contain the answer; the model should decline.\n"
            f"ANSWER KEY:\n{case.correct_answer.strip() or '(none: the information is not in the documents)'}"
        )
    else:
        parts.append(f"ANSWER KEY:\n{case.correct_answer.strip() or '(none given)'}")
        if case.keywords and "; ".join(case.keywords) != case.correct_answer.strip():
            parts.append("KEY VALUES:\n" + "\n".join(f"- {k}" for k in case.keywords))
    parts.append(f"MODEL ANSWER:\n{case.model_answer.strip() or '(empty)'}")
    return [
        {"role": "system", "content": JUDGE_PROMPT},
        {"role": "user", "content": "\n\n".join(parts)},
    ]


_THINK = re.compile(r"<think>.*?</think>", re.DOTALL | re.IGNORECASE)
_FENCE = re.compile(r"```(?:json)?\s*(.*?)```", re.DOTALL | re.IGNORECASE)
_VERDICT_FIELD = re.compile(r'"verdict"\s*:\s*"?\s*(pass|partial|fail)\b', re.IGNORECASE)
_VERDICT_LINE = re.compile(r"(?im)^\W*verdict\W*[:=]\W*(pass|partial|fail)\b")


def _json_objects(text: str) -> Iterator[str]:
    """Every balanced ``{...}`` span in ``text``, in order (string-aware, so braces in reasoning don't confuse it)."""
    depth, start, in_str, esc = 0, -1, False, False
    for i, ch in enumerate(text):
        if in_str:
            if esc:
                esc = False
            elif ch == "\\":
                esc = True
            elif ch == '"':
                in_str = False
            continue
        if ch == '"':
            in_str = True
        elif ch == "{":
            if depth == 0:
                start = i
            depth += 1
        elif ch == "}" and depth:
            depth -= 1
            if depth == 0:
                yield text[start:i + 1]


_FACT_STATUS = ("present", "wrong", "missing")


def _squash(text: str) -> str:
    """Case, spacing, punctuation and markdown carry no meaning when checking that a quote is really in an answer."""
    return re.sub(r"\W+", "", (text or "").lower())


def _quote_in(evidence: str, answer: str) -> bool:
    """Is the judge's quoted evidence really inside the model answer? (every ``...``-separated piece must be)."""
    pieces = [p for p in re.split(r"\.\.\.|…", evidence or "") if _squash(p)]
    haystack = _squash(answer)
    return bool(pieces) and all(_squash(p) in haystack for p in pieces)


def _read_facts(raw_facts: Any) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for f in raw_facts if isinstance(raw_facts, list) else []:
        if not isinstance(f, dict):
            continue
        status = str(f.get("status") or "").strip().lower()
        if str(f.get("fact") or "").strip() and status in _FACT_STATUS:
            out.append({"fact": str(f["fact"]).strip(), "status": status, "evidence": str(f.get("evidence") or "").strip()})
    return out


def _rubric(facts: list[dict[str, Any]], extra_wrong: bool, answer: str) -> tuple[Verdict, str]:
    """Verdict implied by the judge's own checklist, plus the checklist as readable text.

    The judge decides, per fact, whether the answer states it; this only adds them up. A "present" claim whose quoted
    evidence is not actually in the model answer is not trusted (a small judge inventing a quote is the failure to catch).
    """
    lines: list[str] = []
    present = 0
    for f in facts:
        ok = f["status"] == "present"
        if ok and not _quote_in(f["evidence"], answer):
            f["status"] = "unverified"
            ok = False
            lines.append(f"? {f['fact']} — the judge's quote {f['evidence']!r} is not in the answer")
        elif ok:
            lines.append(f"✔ {f['fact']} — “{f['evidence']}”")
        else:
            lines.append(f"✘ {f['fact']} ({f['status']})" + (f": “{f['evidence']}”" if f["evidence"] else ""))
        present += int(ok)
    if present == len(facts) and not extra_wrong:
        verdict: Verdict = "pass"
    elif present == 0:
        verdict = "fail"
    else:
        verdict = "partial"
    if extra_wrong:
        lines.append("✘ the answer adds a claim that contradicts the key")
    return verdict, "\n".join(lines)


def parse_judge_reply(raw: str, case: JudgeCase | None = None) -> JudgeResult:
    """Parse a judge reply. Accepts a JSON object (optionally fenced / after <think>) or an explicit ``verdict:`` field.

    With ``case`` and a usable per-fact checklist in the reply, the verdict is the one the checklist implies (see
    :func:`_rubric`). Never infers a verdict from free prose: an unparseable reply returns an empty verdict with ``error`` set.
    """
    text = _THINK.sub("", raw or "").strip()
    if not text:
        return JudgeResult(error="the judge returned an empty reply", raw=raw or "")
    candidates = [m.group(1) for m in _FENCE.finditer(text)] + [text]
    for blob in candidates:
        for span in _json_objects(blob):
            try:
                data = json.loads(span)
            except json.JSONDecodeError:
                continue
            if not isinstance(data, dict):
                continue
            verdict = str(data.get("verdict") or "").strip().lower()
            facts = _read_facts(data.get("facts"))
            use_rubric = bool(facts) and case is not None and not case.expect_abstain
            if verdict not in VERDICTS and not use_rubric:
                continue
            try:
                conf = float(data["confidence"]) if data.get("confidence") is not None else None
            except (TypeError, ValueError):
                conf = None
            reasoning = str(data.get("reasoning") or "").strip()
            if use_rubric:
                implied, checklist = _rubric(facts, bool(data.get("extra_wrong")), case.model_answer)  # type: ignore[union-attr]
                if verdict in VERDICTS and verdict != implied:
                    reasoning = f"{reasoning}\n(the judge first said {verdict}; its checklist below gives {implied})".strip()
                verdict = implied
                reasoning = f"{reasoning}\n{checklist}".strip()
            return JudgeResult(
                verdict=verdict,  # type: ignore[arg-type]
                reasoning=reasoning,
                confidence=None if conf is None else max(0.0, min(1.0, conf)),
                raw=raw,
                facts=facts,
            )
    m = _VERDICT_FIELD.search(text) or _VERDICT_LINE.search(text)
    if m:
        return JudgeResult(verdict=m.group(1).lower(), reasoning=text[:600], raw=raw)  # type: ignore[arg-type]
    return JudgeResult(error=f"could not read a verdict from the judge reply: {text[:160]!r}", raw=raw)


def judge_one(chat: ChatFn, case: JudgeCase, *, attempts: int = 2) -> JudgeResult:
    """Ask the judge about one case. A malformed reply is retried once with a reminder; errors never become verdicts."""
    messages = build_judge_messages(case)
    last = JudgeResult(error="judge was not called")
    for attempt in range(max(attempts, 1)):
        try:
            raw = chat(messages)
        except Exception as exc:  # noqa: BLE001 - provider errors vary; the text is what the user needs
            return JudgeResult(error=f"judge call failed: {exc}")
        last = parse_judge_reply(raw, case)
        if last.verdict:
            return last
        if attempt + 1 < attempts:
            messages = [*build_judge_messages(case), {"role": "assistant", "content": raw},
                        {"role": "user", "content": 'Reply with ONE JSON object only: {"facts": [...], "extra_wrong": false, "reasoning": "...", "verdict": "pass|partial|fail", "confidence": 0.0}'}]
    return last


# ── Judge backends: any provider row ──────────────────────────────────────────


class JudgeUnavailable(RuntimeError):
    """The chosen provider cannot judge right now (unknown row, missing GGUF, load failure)."""


@dataclass
class LoadedJudge:
    """A provider that is loaded and ready: ``chat`` answers judge prompts."""

    chat: ChatFn
    provider_id: str
    model: str            # what is recorded next to every verdict
    label: str
    concurrent: bool      # API providers serve parallel requests


def list_judge_providers() -> list[dict[str, Any]]:
    """Provider rows that can act as judge, helper seat first. No keys are ever returned."""
    from finetune_studio.models.helper import (
        annotate_provider,
        get_helper_provider_id,
        missing_gguf_for_provider,
    )
    from finetune_studio.models.manager import get_manager

    seat = get_helper_provider_id()
    rows = []
    for row in get_manager().list_providers():
        row = annotate_provider(row)
        local = row.get("kind") == "local_gguf"
        rows.append({
            "id": row["id"],
            "label": row.get("label") or row.get("name") or row["id"],
            "kind": row.get("kind", ""),
            "model_id": row.get("model_id", ""),
            "is_helper_seat": row["id"] == seat,
            "local": local,
            # a local judge whose GGUF was deleted: pickers disable it instead of failing at judge time
            "file_missing": bool(local and missing_gguf_for_provider(row["id"])),
        })
    rows.sort(key=lambda r: (not r["is_helper_seat"], r["label"].lower()))
    return rows


def default_judge_provider_id() -> str:
    """The configured default judge, else the helper seat."""
    from finetune_studio.models.helper import get_helper_provider_id
    from finetune_studio.webui.routes.settings import get_test_settings

    return get_test_settings()["judge_provider_id"] or get_helper_provider_id()


@contextmanager
def open_judge(provider_id: str) -> Iterator[LoadedJudge]:
    """Load ``provider_id`` as the judge, yield it, and free the GPU afterwards.

    A local GGUF judge replaces whatever model is resident (the one under test was already run and saved), so run
    and judge never compete for the card; an API provider uses no VRAM and leaves resident models alone.
    """
    from finetune_studio.models.helper import (
        annotate_provider,
        missing_gguf_for_provider,
    )
    from finetune_studio.models.manager import get_manager

    mgr = get_manager()
    row = mgr.get_provider(provider_id)
    if row is None:
        raise JudgeUnavailable(f"unknown provider '{provider_id}'")
    row = annotate_provider(row)
    local = row.get("kind") == "local_gguf"
    if local:
        missing = missing_gguf_for_provider(provider_id)
        if missing:
            raise JudgeUnavailable(f"judge model file not found on disk: {missing} (download it again in Model library, or pick another judge)")
        from finetune_studio.data.rag_portable.model_cache import release_rag_models
        from finetune_studio.models.llama_loader import unload_all_models

        unload_all_models()
        release_rag_models("test judge load")
    try:
        mgr.load(provider_id)
    except Exception as exc:
        raise JudgeUnavailable(f"could not load judge '{row.get('label') or provider_id}': {exc}") from exc
    try:
        yield LoadedJudge(
            chat=lambda messages: mgr.chat(messages, max_tokens=JUDGE_MAX_TOKENS, temperature=0.0),
            provider_id=provider_id,
            model=str(row.get("model_id") or row.get("name") or provider_id),
            label=str(row.get("label") or provider_id),
            concurrent=not local,
        )
    finally:
        if local:
            mgr.unload()


def judge_many(
    judge: LoadedJudge,
    items: list[tuple[str, JudgeCase]],
    on_result: Callable[[str, JudgeResult], None],
    *,
    should_stop: Callable[[], bool] = lambda: False,
) -> int:
    """Judge ``(key, case)`` pairs, calling ``on_result(key, result)`` as each finishes. Returns how many were attempted.

    Sequential for a local GGUF, a small thread pool for an API provider. ``on_result`` always runs on the calling
    thread, so it may write the DB without extra locking.
    """
    done = 0
    if not judge.concurrent or len(items) < 2:
        for key, case in items:
            if should_stop():
                break
            on_result(key, judge_one(judge.chat, case))
            done += 1
        return done
    with ThreadPoolExecutor(max_workers=JUDGE_PARALLEL_API, thread_name_prefix="fts-judge") as pool:
        for start in range(0, len(items), JUDGE_PARALLEL_API * 4):
            if should_stop():
                break
            batch = items[start:start + JUDGE_PARALLEL_API * 4]
            futures = [(key, pool.submit(judge_one, judge.chat, case)) for key, case in batch]
            for key, fut in futures:
                on_result(key, fut.result())
                done += 1
    return done
