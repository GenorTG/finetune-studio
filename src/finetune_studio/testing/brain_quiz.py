"""A quiz written from the indexed documents: the Smart Brain's own test, no training data needed.

The AI assistant writes one question and answer per sampled passage, spread evenly over the documents, plus questions
the documents do NOT answer (the right reply is to say so). Every generated item passes hard gates or is dropped, never
repaired: an answer may only hold numbers and codes that are in its passage, a question may not give its answer away,
and an "unanswerable" question is checked against the index's top hybrid-search results (with a word-overlap fallback
for direct helper use) so evidence split across documents is not missed. The quiz is saved as a project suite and runs
and judges like any other.
"""
from __future__ import annotations

import json
import random
import re
from collections import defaultdict
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass, field
from typing import Any

from finetune_studio.data.fs.paths import brain_index_dir

GenerateFn = Callable[[str], str]
SearchFn = Callable[[str, int], list[dict[str, Any]]]
ProgressFn = Callable[[dict[str, Any]], None]

SUITE_FILE = "smart-brain-quiz.json"
DEFAULT_QUESTIONS = 40
MAX_QUESTIONS = 200
UNANSWERABLE_SHARE = 0.15
MIN_PASSAGE_CHARS = 150
MAX_PASSAGE_CHARS = 3000
_CLOSEST_PASSAGES = 8
_VERIFY_TOP_K = 10
_MIN_QUESTION_WORDS = 5

_HARD_VALUE = re.compile(r"\d[\d.,:/-]*\d|\d|\b[A-Z]{2,}[-_]?\d[\w-]*\b")
_WORD = re.compile(r"[^\W\d_]{3,}", re.UNICODE)

_ANSWERABLE = """You write one exam question that checks whether someone can find a fact in a company's documents.

DOCUMENT: {filename}
PASSAGE:
{passage}

Reply with one JSON object and nothing else:
{{"question": "...", "answer": "...", "must_contain": ["..."]}}

Rules: ask about one specific fact this passage states. The question must make sense on its own, to someone who has not
seen the passage: name the subject, never say "this document" or "the passage". The answer is short and uses only this
passage. "must_contain" lists the exact values from the passage a correct answer has to include (a number, date, name
or code); use an empty list if the answer is not a specific value. If the passage holds no usable fact, reply {{}}."""

_UNANSWERABLE = """You write one question that a company's documents do NOT answer.

The documents include {filename}, which says:
{passage}

Reply with one JSON object and nothing else: {{"question": "..."}}

Rules: the question sounds like a natural one about this same company or subject, and asks for a specific detail (an
amount, a date, a name) that this text does not give. Make sense on its own; never say "this document"."""

_VERIFY = """Could a careful reader answer this question from these passages? Reply with one JSON object and nothing else: \
{{"answered": true}} or {{"answered": false}}.

Count it as answered when a passage gives the answer directly OR by a clear implication, for example:
- a rule for a whole range, plan or group answers it for each member ("every model in the range has a two-year warranty" answers it for one model of that range);
- "included", "free" or "no charge" answers a question about the cost;
- a general policy answers a question about one case it covers.
Reply false only when no passage gives the asked detail at all.

QUESTION: {question}

PASSAGES:
{passages}"""


class NoIndex(LookupError):
    """The project has no built index with passages to write questions from."""


class NoUsableQuestions(ValueError):
    def __init__(self, message: str, report: BrainQuizReport) -> None:
        super().__init__(message)
        self.report = report


@dataclass
class BrainQuizReport:
    planned: int = 0
    kept: int = 0
    kept_unanswerable: int = 0
    documents: int = 0
    helper_errors: int = 0
    dropped: dict[str, int] = field(default_factory=dict)

    def drop(self, reason: str) -> None:
        self.dropped[reason] = self.dropped.get(reason, 0) + 1

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


def _json_object(reply: str) -> dict[str, Any] | None:
    match = re.search(r"\{.*\}", reply or "", re.DOTALL)
    if not match:
        return None
    try:
        value = json.loads(match.group(0))
    except ValueError:
        return None
    return value if isinstance(value, dict) else None


def _values(text: str) -> set[str]:
    return {v.rstrip(".,:/-").replace(",", "") for v in _HARD_VALUE.findall(text)}


def spread_passages(rows: list[dict[str, str]], count: int, *, seed: int = 7) -> list[dict[str, str]]:
    """Up to ``count`` usable passages, taken in turn from each document so a big file does not fill the quiz."""
    rng = random.Random(seed)
    by_doc: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        if len(row["text"].strip()) >= MIN_PASSAGE_CHARS:
            by_doc[row["document_id"]].append(row)
    for items in by_doc.values():
        rng.shuffle(items)
    out: list[dict[str, str]] = []
    docs = sorted(by_doc)
    while len(out) < count and any(by_doc.values()):
        for doc in docs:
            if by_doc[doc] and len(out) < count:
                out.append(by_doc[doc].pop())
    return out


def closest_passages(question: str, rows: list[dict[str, str]], limit: int = _CLOSEST_PASSAGES) -> list[str]:
    """The passages sharing the most words with the question (no embedder: this runs inside a web request's thread)."""
    words = {w.lower() for w in _WORD.findall(question)}
    scored = sorted(rows, key=lambda r: len(words & {w.lower() for w in _WORD.findall(r["text"])}), reverse=True)
    return [r["text"][:MAX_PASSAGE_CHARS] for r in scored[:limit]]


def answerable_case(row: dict[str, str], reply: str, report: BrainQuizReport) -> dict[str, Any] | None:
    item = _json_object(reply)
    if item is None:
        report.drop("not JSON")
        return None
    question, answer = str(item.get("question") or "").strip(), str(item.get("answer") or "").strip()
    if not question or not answer:
        report.drop("no usable fact")
        return None
    if len(question.split()) < _MIN_QUESTION_WORDS or re.search(r"\b(?:this|the) (?:document|passage|text)\b", question, re.IGNORECASE):
        report.drop("question does not stand alone")
        return None
    passage = row["text"]
    expect = [str(v).strip() for v in (item.get("must_contain") or []) if str(v).strip()]
    if not _values(answer) <= _values(passage) or any(v.lower() not in passage.lower() for v in expect):
        report.drop("answer has a value that is not in the passage")
        return None
    if any(v.lower() in question.lower() for v in expect) or (_values(answer) and _values(answer) <= _values(question)):
        report.drop("question gives the answer away")
        return None
    return {"question": question, "correct_answer": answer, **({"expect": expect} if expect else {}), "category": "documents"}


def unanswerable_case(row: dict[str, str], reply: str, rows: list[dict[str, str]], generate: GenerateFn,
                      report: BrainQuizReport, search: SearchFn | None = None) -> dict[str, Any] | None:
    item = _json_object(reply)
    question = str((item or {}).get("question") or "").strip()
    if len(question.split()) < _MIN_QUESTION_WORDS:
        report.drop("unanswerable: no usable question")
        return None
    passages = ([f"[source: {hit.get('filename') or hit.get('source') or 'document'}]\n{hit.get('text', '')}"
                 for hit in search(question, _VERIFY_TOP_K)] if search is not None
                else closest_passages(question, rows))
    verdict = _json_object(generate(_VERIFY.format(question=question, passages="\n---\n".join(passages))))
    if not verdict or verdict.get("answered") is not False:
        report.drop("unanswerable: the documents may answer it")
        return None
    return {"question": question, "expect_abstain": True, "category": "unanswerable"}


def build_cases(rows: list[dict[str, str]], generate: GenerateFn, *, questions: int = DEFAULT_QUESTIONS, workers: int = 1,
                progress: ProgressFn | None = None, verify: GenerateFn | None = None,
                search: SearchFn | None = None,
                ) -> tuple[list[dict[str, Any]], BrainQuizReport]:
    """Write cases; failed helper calls drop that item, not the quiz. ``verify`` (default: ``generate``) checks each
    unanswerable candidate deterministically (temperature 0 in the app). ``search`` supplies the indexed passages that
    might jointly answer it; the local overlap helper remains available to simple callers without an index.

    """
    verify = verify or generate
    report = BrainQuizReport(documents=len({r["document_id"] for r in rows}))
    n_unknown = max(1, round(questions * UNANSWERABLE_SHARE))
    answer_rows = spread_passages(rows, questions - n_unknown)
    unknown_rows = spread_passages(rows, n_unknown, seed=11)
    report.planned = len(answer_rows) + len(unknown_rows)
    done = 0

    def one(task: tuple[str, dict[str, str]]) -> dict[str, Any] | None:
        kind, row = task
        passage = row["text"][:MAX_PASSAGE_CHARS]
        try:
            if kind == "answer":
                return answerable_case(row, generate(_ANSWERABLE.format(filename=row["filename"], passage=passage)), report)
            reply = generate(_UNANSWERABLE.format(filename=row["filename"], passage=passage))
            return unanswerable_case(row, reply, rows, verify, report, search)
        except Exception:  # noqa: BLE001 - one provider failure must not lose the rest of the quiz
            report.helper_errors += 1
            return None

    tasks = [("answer", r) for r in answer_rows] + [("unknown", r) for r in unknown_rows]
    cases: list[dict[str, Any]] = []
    with ThreadPoolExecutor(max_workers=max(1, workers)) as pool:
        for case in pool.map(one, tasks):
            done += 1
            if case is not None:
                cases.append(case)
            if progress is not None:
                progress({"done": done, "planned": report.planned, "kept": len(cases)})
    report.kept = len(cases)
    report.kept_unanswerable = sum(bool(c.get("expect_abstain")) for c in cases)
    return cases, report


def index_rows(pid: str) -> list[dict[str, str]]:
    path = brain_index_dir(pid) / "chunks.parquet"
    if not path.is_file():
        raise NoIndex("Build the Smart Brain first: the quiz is written from its passages.")
    import pandas as pd

    frame = pd.read_parquet(path, columns=["document_id", "filename", "text"])
    if frame.empty:
        raise NoIndex("The Smart Brain index is empty: build it first.")
    rows = frame.astype(str).to_dict("records")
    from finetune_studio.knowledge.lint import excluded_documents

    # The knowledge check's out-of-date and duplicate files: a quiz built from them would mark the current answer wrong.
    skip = excluded_documents(pid)
    kept = [r for r in rows if r["document_id"] not in skip]
    if not kept:
        raise NoIndex("Every indexed document is marked out of date or duplicate in the Knowledge check.")
    return kept


def build_brain_quiz(pid: str, *, questions: int = DEFAULT_QUESTIONS, generate: GenerateFn | None = None,
                     workers: int | None = None, progress: ProgressFn | None = None,
                     search: SearchFn | None = None) -> dict[str, Any]:
    """Write and save the project's Smart Brain quiz (rebuilding replaces it). Blocking: model calls."""
    from finetune_studio.testing.suite_import import import_suite

    rows = index_rows(pid)
    verify: GenerateFn | None = None
    if generate is None:
        from finetune_studio.data.mining.scheduler import helper_info
        from finetune_studio.data.prep.generator import helper_resolution_error
        from finetune_studio.data.prep.preference import NoGenerator, helper_generate

        generate = helper_generate(max_tokens=400, temperature=0.4, job="quiz")
        if generate is None:
            raise NoGenerator(helper_resolution_error("quiz"))
        verify = helper_generate(max_tokens=400, temperature=0.0, job="quiz")
        workers = workers or int(helper_info("quiz").get("concurrency") or 1)
    if search is None:
        from finetune_studio.smart_brain import BrainIndex

        query = BrainIndex(brain_index_dir(pid)).load()

        def search_index(question: str, top_k: int) -> list[dict[str, Any]]:
            return query.search(question, top_k=top_k, hybrid=True)

        search = search_index
    cases, report = build_cases(rows, generate, questions=questions, workers=workers or 1, progress=progress,
                                verify=verify, search=search)
    if not any(not c.get("expect_abstain") for c in cases):
        raise NoUsableQuestions("No question passed the quality checks, so no quiz was saved.", report)
    meta = import_suite(pid, SUITE_FILE, json.dumps(cases, ensure_ascii=False).encode("utf-8"))
    return {"name": meta["name"], "path": meta["path"], "case_count": meta["case_count"], **report.as_dict()}
