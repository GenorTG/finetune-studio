"""Reworded held-out quiz: the trainer's own held-out facts, each question asked once in words no dataset row uses.

The held-out slice is ``training.data.sft_validation_rows`` (same formatter, groups and seed as the trainer), so every
row sharing a ``pair_id`` is held out together and no trained row is a rewording of a quiz fact. Per held-out fact (one
group; its ``study_kind == "original"`` row when there is one) the data-prep helper writes ONE new question. It passes
``study_set.question_gate`` (keeps the original's numbers, codes and most names, leaks no answer value), must differ
from every question in the dataset and must not be (nearly) the original wording, or the fact is dropped, never
repaired. The answer key is the reviewed answer as it is.

The quiz is stored as a project suite (``testing.suite_import``) named ``heldout-reworded-<dataset>``: the Testing page
lists it as "held-out facts, new wording", runs and judges it like any quiz, and ``study_set.heldout_questions`` keeps
future study sets away from its questions. The auto-suite (training questions word for word) stays the memorisation
check. Grounded rows (retrieved CONTEXT in their system turn) are skipped: a suite question is asked bare.
"""
from __future__ import annotations

import json
import re
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass, field
from typing import Any

from finetune_studio.benchmarks.suite_defs import (
    HELDOUT_REWORDED_PREFIX,
    format_heldout_reworded_label,
)
from finetune_studio.data.prep.qa_validate import normalize_question
from finetune_studio.data.prep.study_set import (
    NEAR_QUIZ_JACCARD,
    _words,
    parse_reply,
    question_gate,
)
from finetune_studio.smart_brain.prompt import CONTEXT_MARKER
from finetune_studio.testing.training_eval import (
    _extract_qa,
    read_dataset_rows,
    resolve_project_dataset,
)

DEFAULT_MAX_QUESTIONS = 200
MAX_QUESTIONS = 1000
_MAX_CONSECUTIVE_ERRORS = 5

GenerateFn = Callable[[str], str]
ProgressFn = Callable[[dict[str, Any]], None]

_PROMPT = """You reword one quiz question so it tests the same fact in new words.

QUESTION: {question}
ANSWER (never put any of it into the question): {answer}

Reply with one JSON object and nothing else:
{{"question": "the QUESTION re-worded: same meaning and the same specific names, numbers and codes; a different
sentence form, word order and formality; it must not reveal anything from the ANSWER"}}"""


class RewordedQuizError(Exception):
    """Base for honest, user-facing build failures."""


class NoHeldoutRows(RewordedQuizError):
    """The dataset's held-out slice has no question the quiz can ask."""


class NoUsableQuestions(RewordedQuizError):
    """Every rewording failed a gate (or every call failed); ``report`` says why."""

    def __init__(self, message: str, report: RewordReport) -> None:
        super().__init__(message)
        self.report = report


@dataclass
class RewordReport:
    heldout_rows: int = 0
    facts: int = 0
    skipped_grounded: int = 0
    calls_failed: int = 0
    kept: int = 0
    dropped: dict[str, int] = field(default_factory=dict)

    def drop(self, reason: str) -> None:
        self.dropped[reason] = self.dropped.get(reason, 0) + 1

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class HeldoutFact:
    key: str
    question: str
    answer: str
    source_id: str = ""
    chunk_idx: int = 0


def _grounded(row: dict[str, Any]) -> bool:
    turns = row.get("conversations") or row.get("messages") or []
    return any(isinstance(t, dict) and str(t.get("from") or t.get("role") or "").lower() == "system"
               and CONTEXT_MARKER in str(t.get("value") or t.get("content") or "") for t in turns) \
        or CONTEXT_MARKER in str(row.get("system") or "")


def heldout_facts(rows: list[dict[str, Any]], report: RewordReport) -> list[HeldoutFact]:
    """One fact per held-out group (``pair_id``, else the question itself), its original row preferred."""
    from finetune_studio.training.data import sft_validation_rows

    groups: dict[str, list[dict[str, Any]]] = {}
    held = sft_validation_rows(rows)
    report.heldout_rows = len(held)
    for row in held:
        if _grounded(row):
            report.skipped_grounded += 1
            continue
        qa = _extract_qa(row)
        if qa is None or not qa[1].strip():
            continue
        key = str(row.get("pair_id") or "") or "q:" + normalize_question(qa[0])
        groups.setdefault(key, []).append(row)
    facts = []
    for key, members in groups.items():
        row = next((r for r in members if r.get("study_kind") == "original"), members[0])
        q, a = _extract_qa(row) or ("", "")
        facts.append(HeldoutFact(key=key, question=q, answer=a.strip(), source_id=str(row.get("source_id") or ""),
                                 chunk_idx=int(row.get("chunk_idx") or 0)))
    report.facts = len(facts)
    return facts


def reworded_question(fact: HeldoutFact, reply: str, dataset_questions: set[str]) -> tuple[str | None, str | None]:
    """(question, None) when the helper's rewording passes every gate, else (None, drop reason)."""
    try:
        data = parse_reply(reply)
    except ValueError:
        return None, "unparseable_reply"
    v = str(data.get("question") or "").strip()
    reason = question_gate(fact.question, fact.answer, v, dataset_questions)
    if reason:
        return None, reason
    words, orig = _words(v), _words(fact.question)
    if words and orig and len(words & orig) / len(words | orig) >= NEAR_QUIZ_JACCARD:
        return None, "question_same_wording"
    return v, None


def build_reworded_cases(rows: list[dict[str, Any]], generate: GenerateFn, *,
                         max_questions: int = DEFAULT_MAX_QUESTIONS, workers: int = 1,
                         progress: ProgressFn | None = None) -> tuple[list[dict[str, Any]], RewordReport]:
    """Suite cases (``suite_import`` native rows) for the dataset's held-out facts, one helper call per fact."""
    report = RewordReport()
    facts = heldout_facts(rows, report)[:max_questions]
    if not facts:
        raise NoHeldoutRows("The dataset's held-out slice has no question to reword (it is too small, or every "
                            "held-out row is grounded in retrieved context). Add pairs, re-export, try again.")
    dataset_questions = {normalize_question(qa[0]) for r in rows if (qa := _extract_qa(r))}

    def work(fact: HeldoutFact) -> tuple[HeldoutFact, str | None]:
        try:
            return fact, generate(_PROMPT.format(question=fact.question, answer=fact.answer))
        except Exception:  # noqa: BLE001 - a failed call drops that fact, counted below
            return fact, None

    cases: list[dict[str, Any]] = []
    consecutive = 0
    with ThreadPoolExecutor(max_workers=max(1, workers)) as pool:
        for done, (fact, reply) in enumerate(pool.map(work, facts), 1):
            if reply is None:
                report.calls_failed += 1
                consecutive += 1
                if consecutive >= _MAX_CONSECUTIVE_ERRORS and not cases:
                    raise NoUsableQuestions(f"The assistant failed {consecutive} calls in a row; no question was "
                                            "reworded.", report)
            else:
                consecutive = 0
                question, reason = reworded_question(fact, reply, dataset_questions)
                if reason:
                    report.drop(reason)
                else:
                    dataset_questions.add(normalize_question(question or ""))
                    cases.append({"name": f"rq{len(cases) + 1:04d}", "category": "heldout_reworded",
                                  "question": question, "correct_answer": fact.answer, "keywords": []})
                    report.kept += 1
            if progress:
                progress({"done": done, "planned": len(facts), "kept": report.kept})
    return cases, report


@dataclass(frozen=True)
class BuiltQuiz:
    name: str
    path: str
    label: str
    case_count: int
    dataset: dict[str, Any]
    report: RewordReport


def suite_stem(dataset_name: str) -> str:
    slug = re.sub(r"[^A-Za-z0-9._-]+", "-", dataset_name).strip("-.")[:60] or "dataset"
    return f"{HELDOUT_REWORDED_PREFIX}-{slug}"


def build_reworded_quiz(pid: str, *, dataset_id: str | None = None, max_questions: int = DEFAULT_MAX_QUESTIONS,
                        generate: GenerateFn | None = None, workers: int | None = None,
                        progress: ProgressFn | None = None) -> BuiltQuiz:
    """Build and save the project's reworded held-out quiz for one dataset (rebuilding replaces it).

    Blocking (model calls): async callers run it with ``asyncio.to_thread``. ``generate=None`` uses the seated helper
    with its parallel-call setting; no loaded helper raises ``preference.NoGenerator`` with the reason. A missing
    dataset raises ``LookupError``.
    """
    from finetune_studio.testing.suite_import import import_suite

    ds = resolve_project_dataset(pid, dataset_id)
    rows, _bad = read_dataset_rows(str(ds["data_path"]))
    if generate is None:
        from finetune_studio.data.mining.scheduler import helper_info
        from finetune_studio.data.prep.generator import helper_resolution_error
        from finetune_studio.data.prep.preference import NoGenerator, helper_generate

        generate = helper_generate(max_tokens=300, temperature=0.7, job="quiz")
        if generate is None:
            raise NoGenerator(helper_resolution_error("quiz"))
        workers = workers or int(helper_info("quiz").get("concurrency") or 1)
    cases, report = build_reworded_cases(rows, generate, max_questions=max_questions, workers=workers or 1,
                                         progress=progress)
    if not cases:
        raise NoUsableQuestions("No reworded question passed the quality checks, so no quiz was saved.", report)
    dataset_name = str(ds.get("name") or ds.get("id") or "dataset")
    stem = suite_stem(dataset_name)
    meta = import_suite(pid, f"{stem}.json", json.dumps(cases, ensure_ascii=False).encode("utf-8"))
    title = stem.removeprefix(HELDOUT_REWORDED_PREFIX + "-")
    return BuiltQuiz(name=meta["name"], path=meta["path"], label=format_heldout_reworded_label(title, len(cases)),
                     case_count=meta["case_count"], dataset={"id": ds.get("id"), "name": dataset_name}, report=report)
