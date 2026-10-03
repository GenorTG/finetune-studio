"""Training-dataset health check and duplicate removal (WebUI training page).

WHAT THIS FILE DOES
===================
Reads one dataset JSONL and reports, in plain language, what will break or
weaken a training run: unreadable lines, rows the trainer silently skips,
missing answers, duplicate examples, the same question taught with different
answers, very short answers, and a held-out split too small to evaluate on.
Every issue names the 1-based line numbers it found (first few), so the user
can open them in the Data Editor.

Rows are judged through ``training.data.format_for_sft`` — the exact function
training uses — so "trainable" here means trainable there.

Deliberately NOT checked (see the 2026-10-03 UI coverage audit): language
"balance" (a monolingual dataset is normal) and "I don't know" answers (that is
the behaviour we want for out-of-scope questions).
"""

from __future__ import annotations

import copy
import json
import re
from pathlib import Path
from typing import Any

from finetune_studio.training.data import format_for_sft

MAX_LINES_SHOWN = 5
SMALL_DATASET = 20  # below this the 10% held-out split is a handful of rows
SHORT_ANSWER_WORDS = 3
LONG_ANSWER_WORDS = 500


def _norm(text: str) -> str:
    return re.sub(r"\s+", " ", str(text)).strip().lower()


def _issue(code: str, severity: str, title: str, detail: str, lines: list[int],
           count: int | None = None, action: str | None = None) -> dict[str, Any]:
    out: dict[str, Any] = {"code": code, "severity": severity, "title": title, "detail": detail,
                           "count": len(lines) if count is None else count,
                           "lines": lines[:MAX_LINES_SHOWN]}
    if action:
        out["action"] = action
    return out


def _example_key(messages: list[dict]) -> str:
    return json.dumps([(m.get("role"), _norm(m.get("content", ""))) for m in messages])


def _read_rows(path: Path) -> list[tuple[int, str, Any]]:
    """``(line_no, raw_line, parsed_or_None)`` for every non-blank line."""
    rows = []
    with open(path, encoding="utf-8", errors="replace") as f:
        for n, line in enumerate(f, 1):
            raw = line.rstrip("\n")
            if not raw.strip():
                continue
            try:
                rows.append((n, raw, json.loads(raw)))
            except json.JSONDecodeError:
                rows.append((n, raw, None))
    return rows


def check_dataset(path: str | Path) -> dict[str, Any]:
    """Health report for one dataset file. See module docstring."""
    rows = _read_rows(Path(path))
    invalid, untrainable, no_answer, dupes, short, long_ = [], [], [], [], [], []
    seen: dict[str, int] = {}
    answers_by_q: dict[str, dict[str, int]] = {}
    systems: set[str] = set()
    answer_words: list[int] = []
    trainable = 0

    for n, _raw, item in rows:
        if not isinstance(item, dict):
            invalid.append(n)
            continue
        formatted = format_for_sft([copy.deepcopy(item)])
        if not formatted:
            untrainable.append(n)
            continue
        msgs = formatted[0]["messages"]
        trainable += 1
        answers = [m for m in msgs if m.get("role") == "assistant"]
        if not answers or not all(str(m.get("content", "")).strip() for m in answers):
            no_answer.append(n)
            continue
        key = _example_key(msgs)
        if key in seen:
            dupes.append(n)
            continue
        seen[key] = n
        systems.update(_norm(m.get("content", ""))[:200] for m in msgs if m.get("role") == "system")
        words = len(str(answers[-1].get("content", "")).split())
        answer_words.append(words)
        if words < SHORT_ANSWER_WORDS:
            short.append(n)
        elif words > LONG_ANSWER_WORDS:
            long_.append(n)
        users = [m for m in msgs if m.get("role") == "user"]
        if users:
            q = _norm(users[-1].get("content", ""))
            answers_by_q.setdefault(q, {}).setdefault(_norm(answers[-1].get("content", "")), n)

    conflicts = sorted(ln for by_answer in answers_by_q.values() if len(by_answer) > 1
                       for ln in by_answer.values())
    n_conflict_q = sum(1 for by_answer in answers_by_q.values() if len(by_answer) > 1)
    holdout = trainable - int(trainable * 0.9)

    issues = []
    if invalid:
        issues.append(_issue("invalid_json", "error", "Unreadable lines",
                             "These lines are not valid JSON objects and are skipped by training.", invalid))
    if untrainable:
        issues.append(_issue("untrainable", "error", "Rows training will skip",
                             "No messages, conversations, text or prompt+completion fields, so the trainer "
                             "drops them silently.", untrainable))
    if no_answer:
        issues.append(_issue("no_answer", "error", "Missing or empty answers",
                             "Examples without an assistant answer teach the model to reply with nothing.",
                             no_answer))
    if dupes:
        issues.append(_issue("duplicates", "warning", "Duplicate examples",
                             "Exact repeats are trained on twice and leak into the held-out split, "
                             "inflating evaluation scores.", dupes, action="dedup"))
    if conflicts:
        issues.append(_issue("conflicting_answers", "warning", "Same question, different answers",
                             "The model is taught contradictory answers to one question; keep the right one "
                             "in the Data Editor.", conflicts, count=n_conflict_q))
    if trainable and trainable < SMALL_DATASET:
        issues.append(_issue("small_dataset", "warning", "Very small dataset",
                             f"Only {trainable} usable examples; the held-out split is {holdout}, too few for "
                             "evaluation or early stopping to mean much.", [], count=trainable))
    if answer_words and len(short) > len(answer_words) * 0.1:
        issues.append(_issue("short_answers", "info", "Many very short answers",
                             f"{len(short)} answers are under {SHORT_ANSWER_WORDS} words; the model will learn "
                             "to answer tersely.", short))
    if long_:
        issues.append(_issue("long_answers", "info", "Very long answers",
                             f"Answers over {LONG_ANSWER_WORDS} words may be cut off by Max sequence.", long_))
    if len(systems) > 3:
        issues.append(_issue("system_prompts", "info", "Several different system prompts",
                             f"{len(systems)} distinct system prompts; the model may not settle on one persona.",
                             [], count=len(systems)))

    severities = {i["severity"] for i in issues}
    verdict = "errors" if "error" in severities else "warnings" if "warning" in severities else "ok"
    return {
        "examples": len(rows), "trainable": trainable, "holdout": holdout, "verdict": verdict,
        "issues": issues,
        "stats": {
            "avg_answer_words": round(sum(answer_words) / len(answer_words), 1) if answer_words else 0,
            "min_answer_words": min(answer_words, default=0),
            "max_answer_words": max(answer_words, default=0),
        },
    }


def dedupe_dataset(src: str | Path, dst: str | Path) -> tuple[int, int]:
    """Copy ``src`` to ``dst`` without exact-duplicate examples. Returns ``(kept, removed)``.

    Keeps the first occurrence and original line text; lines that are not
    trainable examples are copied unchanged (fixing them is not dedup's job).
    """
    seen: set[str] = set()
    kept, removed = [], 0
    for _n, raw, item in _read_rows(Path(src)):
        formatted = format_for_sft([copy.deepcopy(item)]) if isinstance(item, dict) else []
        if formatted:
            key = _example_key(formatted[0]["messages"])
            if key in seen:
                removed += 1
                continue
            seen.add(key)
        kept.append(raw)
    Path(dst).write_text("".join(line + "\n" for line in kept), encoding="utf-8")
    return len(kept), removed
