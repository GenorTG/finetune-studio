"""Study set: each approved fact written several ways, for SFT that has to recall facts.

Evidence (docs/audit/2026-10-24-training-research.md §3): what makes a fine-tuned model answer a fact asked in new
words is how many DIFFERENT forms of the fact it trained on, not epochs over one wording (Physics of LMs 3.1: 9.7 % ->
96.6 % with varied rewrites; Active Reading: repetition 15.9, paraphrase 25.7, varied study material 63-66). Korvane
§21: adding three reworded questions per pair lifted reworded-quiz recall from 15 to 48 of 102.

Per approved pair, one helper call writes (each kind optional):

* ``reworded``  — the question in other words; the reviewed answer is kept as is.
* ``restated``  — the reviewed answer said differently, asked with the original question.
* ``passage``   — the fact as a short document passage, with a question asking about its subject.

Every generated text passes hard gates or is dropped, never repaired: a question keeps the original question's numbers
and codes and most of its names and leaks no answer value; an answer or passage keeps every number, date and code of the
reviewed answer and adds none that is not in the answer, the question or the pair's source passage. A question that is
(nearly) one of the project's held-out quiz questions is dropped so the quiz stays a real test. Rows carry
``study_kind`` so a dataset shows what was generated. The rows are what the helper wrote: review the dataset like any
other before trusting a run.
"""
from __future__ import annotations

import json
import re
from collections.abc import Callable, Iterable
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass, field
from typing import Any

from finetune_studio.data.mining.statements import hard_tokens, join_thousands
from finetune_studio.data.prep.export import deduplicate_qa_pairs
from finetune_studio.data.prep.qa_validate import normalize_question
from finetune_studio.training.data import clean_answer_for_training

KINDS = ("reworded", "restated", "passage")
DEFAULT_REWORDED = 2
DEFAULT_RESTATED = 1
MAX_PER_KIND = 5
# A generated question this close (token Jaccard) to a held-out quiz question is dropped.
NEAR_QUIZ_JACCARD = 0.8
# A restated answer this close to the reviewed one adds no new form.
MAX_ANSWER_JACCARD = 0.9
_MIN_NAME_SHARE = 0.7
_MAX_CONSECUTIVE_ERRORS = 5

GenerateFn = Callable[[str], str]
ProgressFn = Callable[[dict[str, Any]], None]

_PROMPT = """You write a study set so a language model can learn one fact from several angles.

QUESTION: {question}
REVIEWED ANSWER: {answer}
SOURCE PASSAGE (for reference only): {chunk}

Reply with one JSON object and nothing else:
{{"questions": [{n_questions} different re-wordings of the QUESTION: same meaning and the same specific names, numbers and
codes, different sentence form, word order and formality; never put facts from the answer into a question],
"answers": [{n_answers} different ways to state the REVIEWED ANSWER: keep every number, date, code and name it contains,
change the wording and sentence structure, add nothing that is not in the answer or the source passage],
"passage_question": "{passage_q_hint}",
"passage": "{passage_hint}"}}"""

_CAP = re.compile(r"(?<=[a-z,;:] )[A-Z][A-Za-zÀ-ɏ\-]{2,}")
_THINK = re.compile(r"<think>.*?</think>", re.DOTALL | re.IGNORECASE)


class StudySetError(Exception):
    """Base for honest, user-facing build failures."""


class NoApprovedPairs(StudySetError):
    """The project has no approved pair to build from."""


class NoUsableRows(StudySetError):
    """Every generated text failed a gate (or every call failed); ``report`` says why."""

    def __init__(self, message: str, report: StudyReport) -> None:
        super().__init__(message)
        self.report = report


@dataclass
class StudyReport:
    pairs: int = 0
    calls_failed: int = 0
    kept: dict[str, int] = field(default_factory=lambda: dict.fromkeys(KINDS, 0))
    dropped: dict[str, int] = field(default_factory=dict)
    pairs_without_new_row: int = 0

    def drop(self, reason: str) -> None:
        self.dropped[reason] = self.dropped.get(reason, 0) + 1

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class StudyOptions:
    reworded: int = DEFAULT_REWORDED
    restated: int = DEFAULT_RESTATED
    passage: bool = True

    def __post_init__(self) -> None:
        for name in ("reworded", "restated"):
            n = getattr(self, name)
            if not 0 <= n <= MAX_PER_KIND:
                raise ValueError(f"{name} must be 0..{MAX_PER_KIND}, got {n}")
        if not (self.reworded or self.restated or self.passage):
            raise ValueError("pick at least one kind of study row")


_DIGITS = re.compile(r"\d+")


def values(text: str) -> frozenset[str]:
    """Numbers, dates, codes (``statements.hard_tokens``) plus every digit run, single digits included."""
    return frozenset(hard_tokens(text) | set(_DIGITS.findall(join_thousands(text))))


def _words(text: str) -> set[str]:
    return set(normalize_question(text).split())


def _jaccard(a: set[str], b: set[str]) -> float:
    return len(a & b) / len(a | b) if a and b else 0.0


def _names(text: str) -> set[str]:
    return {t.lower() for t in _CAP.findall(text)}


def question_gate(original_q: str, answer: str, variant: str, seen: set[str]) -> str | None:
    """Why a reworded question is dropped, None when it is kept."""
    v = variant.strip()
    if not 12 <= len(v) <= 400:
        return "question_length"
    if normalize_question(v) in seen:
        return "question_duplicate"
    q_tokens, v_tokens = values(original_q), values(v)
    if not q_tokens <= v_tokens:
        return "question_lost_value"
    names = _names(original_q)
    if names and sum(n in v.lower() for n in names) < _MIN_NAME_SHARE * len(names):
        return "question_lost_name"
    if (v_tokens - q_tokens) & values(answer):
        return "question_leaks_answer"
    return None


def answer_gate(answer: str, candidate: str, allowed: frozenset[str], *, passage: bool = False) -> str | None:
    """Why a restated answer / passage is dropped, None when it is kept."""
    c = candidate.strip()
    if not 2 <= len(c) <= (1500 if passage else 1000):
        return "answer_length"
    got = values(c)
    if not values(answer) <= got:
        return "answer_lost_value"
    if got - allowed:
        return "answer_new_value"
    if not passage and _jaccard(_words(answer), _words(c)) >= MAX_ANSWER_JACCARD:
        return "answer_same_as_reviewed"
    return None


def near_quiz(question: str, quiz: Iterable[tuple[str, set[str]]]) -> bool:
    n, t = normalize_question(question), _words(question)
    return any(n == qn or _jaccard(t, qt) >= NEAR_QUIZ_JACCARD for qn, qt in quiz)


def parse_reply(reply: str) -> dict[str, Any]:
    """The JSON object in a helper reply (thinking blocks stripped); ValueError when there is none."""
    text = _THINK.sub("", reply or "")
    m = re.search(r"\{.*\}", text, re.DOTALL)
    if not m:
        raise ValueError("no JSON object in the reply")
    return json.loads(m.group(0))  # "{...}" parses to a dict or raises


def _strings(value: Any) -> list[str]:
    return [str(x).strip() for x in value if isinstance(x, str) and x.strip()] if isinstance(value, list) else []


def prompt_for(item: dict[str, Any], opts: StudyOptions) -> str:
    want_passage = opts.passage
    return _PROMPT.format(
        question=item["question"], answer=item["answer"], chunk=str(item.get("chunk_text") or "(none)")[:3000],
        n_questions=opts.reworded, n_answers=opts.restated,
        passage_q_hint=("a question asking what the documents say about the subject of the fact" if want_passage else ""),
        passage_hint=("2-4 sentences stating the fact as it would read in a document, keeping every number, date, code "
                      "and name of the reviewed answer" if want_passage else ""),
    )


def _row(question: str, answer: str, item: dict[str, Any], kind: str) -> dict[str, Any]:
    return {"conversations": [{"from": "human", "value": question}, {"from": "gpt", "value": answer}],
            "source_id": item.get("source_id") or "", "chunk_idx": item.get("chunk_idx") or 0,
            "pair_id": item.get("id") or "", "study_kind": kind}


def study_rows_for(item: dict[str, Any], reply: str, opts: StudyOptions, quiz: list[tuple[str, set[str]]],
                   report: StudyReport) -> list[dict[str, Any]]:
    """The gated new rows for one pair from the helper's reply (the original row is added by the caller)."""
    data = parse_reply(reply)
    q, a = item["question"], item["answer"]
    allowed = values(a) | values(q) | values(str(item.get("chunk_text") or ""))
    seen = {normalize_question(q)}
    rows: list[dict[str, Any]] = []

    def keep_question(text: str, gate_reason: str | None) -> bool:
        if gate_reason:
            report.drop(gate_reason)
            return False
        if near_quiz(text, quiz):
            report.drop("near_heldout_quiz")
            return False
        seen.add(normalize_question(text))
        return True

    for v in _strings(data.get("questions"))[: opts.reworded]:
        if keep_question(v, question_gate(q, a, v, seen)):
            rows.append(_row(v, a, item, "reworded"))
            report.kept["reworded"] += 1
    answers_seen = {normalize_question(a)}
    for v in _strings(data.get("answers"))[: opts.restated]:
        reason = answer_gate(a, v, allowed) or ("answer_duplicate" if normalize_question(v) in answers_seen else None)
        if reason:
            report.drop(reason)
            continue
        answers_seen.add(normalize_question(v))
        rows.append(_row(q, v, item, "restated"))
        report.kept["restated"] += 1
    if opts.passage:
        pq = str(data.get("passage_question") or "").strip()
        passage = str(data.get("passage") or "").strip()
        reason = answer_gate(a, passage, allowed, passage=True)
        if not reason and not 12 <= len(pq) <= 400:
            reason = "question_length"
        if not reason and values(pq) & (values(a) - values(q)):
            reason = "question_leaks_answer"
        if reason:
            report.drop(reason)
        elif keep_question(pq, None):
            rows.append(_row(pq, passage, item, "passage"))
            report.kept["passage"] += 1
    return rows


def build_study_rows(items: list[dict[str, Any]], generate: GenerateFn, opts: StudyOptions, *,
                     avoid_questions: Iterable[str] = (), workers: int = 1,
                     progress: ProgressFn | None = None) -> tuple[list[dict[str, Any]], StudyReport]:
    """Original + gated study rows for every approved pair; ``workers`` helper calls run at once."""
    pairs = []
    for it in deduplicate_qa_pairs(items):
        answer = clean_answer_for_training(str(it.get("answer") or "")).strip()
        if answer:
            pairs.append({**it, "answer": answer})
    report = StudyReport(pairs=len(pairs))
    quiz = [(normalize_question(q), _words(q)) for q in avoid_questions if q.strip()]
    consecutive = 0

    def work(item: dict[str, Any]) -> tuple[dict[str, Any], str | None]:
        try:
            return item, generate(prompt_for(item, opts))
        except Exception:  # noqa: BLE001 - a failed call drops that pair's study rows, counted below
            return item, None

    rows: list[dict[str, Any]] = []
    with ThreadPoolExecutor(max_workers=max(1, workers)) as pool:
        for done, (item, reply) in enumerate(pool.map(work, pairs), 1):
            new: list[dict[str, Any]] = []
            if reply is None:
                report.calls_failed += 1
                consecutive += 1
            else:
                try:
                    new = study_rows_for(item, reply, opts, quiz, report)
                    consecutive = 0
                except (ValueError, json.JSONDecodeError):
                    report.drop("unparseable_reply")
            if consecutive >= _MAX_CONSECUTIVE_ERRORS and not any(report.kept.values()):
                raise NoUsableRows(f"The helper failed {consecutive} calls in a row; nothing was generated.", report)
            report.pairs_without_new_row += not new
            rows += [_row(item["question"], item["answer"], item, "original"), *new]
            if progress:
                progress({"done": done, "planned": len(pairs), "kept": dict(report.kept)})
    return rows, report


# ── project build: approved pairs -> study rows -> registered SFT dataset ─────────────────────────

def heldout_questions(pid: str) -> list[str]:
    """Questions of the project's own quizzes (Testing page suites): generated questions must stay away from them."""
    from finetune_studio.testing.suite_import import suites_dir

    out: list[str] = []
    folder = suites_dir(pid)
    for path in sorted(folder.glob("*.json")) if folder.is_dir() else []:
        try:
            cases = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            continue
        out += [str(c.get("question") or "") for c in cases if isinstance(c, dict)] if isinstance(cases, list) else []
    return out


@dataclass(frozen=True)
class BuiltStudySet:
    dataset: dict[str, Any]
    path: str
    rows: int
    report: StudyReport


def build_study_dataset(pid: str, opts: StudyOptions, *, generate: GenerateFn | None = None, workers: int | None = None,
                        progress: ProgressFn | None = None) -> BuiltStudySet:
    """Build ``<pid>-study.jsonl`` (sharegpt, originals + study rows) from the approved pairs and register it.

    Blocking (model calls): async callers run it with ``asyncio.to_thread``. ``generate=None`` uses the seated helper
    with its parallel-call setting (an API helper: Settings -> Helper model; a local GGUF: one call at a time).
    """
    from finetune_studio import db
    from finetune_studio.data.fs import qa as qa_fs
    from finetune_studio.data.prep.dataset_build import register_dataset_file

    items = qa_fs.list_qa_pairs(pid, status="approved")
    if not items:
        raise NoApprovedPairs("No approved Q&A pairs yet. Review and approve pairs on the Pairs page first.")
    if generate is None:
        from finetune_studio.data.mining.scheduler import helper_info
        from finetune_studio.data.prep.generator import helper_resolution_error
        from finetune_studio.data.prep.preference import NoGenerator, helper_generate

        generate = helper_generate(max_tokens=1200, temperature=0.7)
        if generate is None:
            raise NoGenerator(helper_resolution_error())
        workers = workers or int(helper_info().get("concurrency") or 1)
    rows, report = build_study_rows(items, generate, opts, avoid_questions=heldout_questions(pid),
                                    workers=workers or 1, progress=progress)
    if not any(report.kept.values()):
        raise NoUsableRows("The helper's texts did not pass a single quality check, so no study rows were added.", report)
    body = "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows)
    name = (db.get_project(pid) or {}).get("name") or pid
    persisted = register_dataset_file(pid, f"{pid}-study.jsonl", body, f"{name} · study set · {{rows}} rows",
                                      source="data-prep-study")
    return BuiltStudySet(dataset=persisted.dataset, path=str(persisted.path), rows=persisted.rows, report=report)
