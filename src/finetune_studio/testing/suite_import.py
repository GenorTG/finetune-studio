"""Import a quiz the user brought (JSON or JSONL) as a project test suite, so the Testing page can run and judge it.

Accepted rows (JSON list, ``{"cases": [...]}`` or one JSON object per line). Two spellings are understood:
  * native:  ``{"name", "question", "correct_answer", "keywords": [...], "category"}``
  * short:   ``{"id", "q", "expect": ["value", ...]}`` — a correct answer holds EVERY expected value (the judge reads them as the answer key)
Add ``"expect_abstain": true`` for a question the documents do not answer: the right reply is "not covered", so a decline is correct
and a confident answer is not. A short row with neither ``expect`` nor ``correct_answer`` must say so explicitly; it is never guessed.
"""
from __future__ import annotations

import json
import re
from pathlib import Path

from finetune_studio.data.fs.paths import project_dir

MAX_CASES = 5000


class SuiteImportError(ValueError):
    """The uploaded file is not a usable suite (message is shown to the user)."""


def suites_dir(pid: str) -> Path:
    return project_dir(pid) / "suites"


def _rows(raw: bytes) -> list:
    text = raw.decode("utf-8-sig", errors="replace").strip()
    if not text:
        raise SuiteImportError("the file is empty")
    try:
        data = json.loads(text)
    except ValueError:
        data = []
        for no, line in enumerate(text.splitlines(), 1):
            if not line.strip():
                continue
            try:
                data.append(json.loads(line))
            except ValueError as e:
                raise SuiteImportError(f"line {no} is not valid JSON ({e})") from e
    if isinstance(data, dict):
        data = data.get("cases")
    if not isinstance(data, list) or not data:
        raise SuiteImportError("expected a JSON list of questions (or {\"cases\": [...]}, or one JSON object per line)")
    return data


def _case(item: object, index: int) -> dict:
    if not isinstance(item, dict):
        raise SuiteImportError(f"row {index} is not an object")
    question = str(item.get("question") or item.get("q") or "").strip()
    if not question:
        raise SuiteImportError(f"row {index} has no question")
    abstain = bool(item.get("expect_abstain"))
    expect = item.get("keywords") or item.get("expect") or []
    if not isinstance(expect, list):
        raise SuiteImportError(f"row {index}: expect/keywords must be a list")
    expect = [str(e) for e in expect if str(e).strip()]
    answer = str(item.get("correct_answer") or "").strip()
    if not abstain and not expect and not answer:
        raise SuiteImportError(f"row {index} ({question[:50]!r}) has no expect/keywords/correct_answer; "
                               'add "expect_abstain": true if the documents do not answer it')
    if abstain:
        answer = answer or "The documents do not say."
    elif not answer:
        answer = "; ".join(expect)  # the judge reads it as the answer key (and sees the values listed as key values)
    name = str(item.get("name") or item.get("id") or f"q{index:04d}")
    return {"name": name, "category": str(item.get("category") or ("unanswerable" if abstain else "quiz")), "question": question,
            "correct_answer": answer, "keywords": expect, "expect_abstain": abstain}


def import_suite(pid: str, filename: str, raw: bytes) -> dict:
    """Validate, normalise and store the file as ``<project>/suites/<name>.json``; returns its description."""
    rows = _rows(raw)
    if len(rows) > MAX_CASES:
        raise SuiteImportError(f"{len(rows)} questions is more than the {MAX_CASES} limit")
    cases = [_case(item, i) for i, item in enumerate(rows, 1)]
    names = [c["name"] for c in cases]
    if len(set(names)) != len(names):
        raise SuiteImportError("question ids/names must be unique (duplicates: "
                               + ", ".join(sorted({n for n in names if names.count(n) > 1})[:5]) + ")")
    stem = re.sub(r"[^A-Za-z0-9._-]+", "-", Path(filename or "quiz").stem).strip("-.") or "quiz"
    out = suites_dir(pid)
    out.mkdir(parents=True, exist_ok=True)
    path = out / f"{stem}.json"
    path.write_text(json.dumps(cases, indent=2, ensure_ascii=False), encoding="utf-8")
    return {"name": stem, "path": str(path), "case_count": len(cases), "abstain_cases": sum(c["expect_abstain"] for c in cases)}
