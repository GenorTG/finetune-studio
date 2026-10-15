"""CRUD for `case_judgements` — every opinion on whether a model answer was correct.

A test run stores the raw transcript of each case with no verdict. Verdicts arrive afterwards, from an AI judge
(any provider row) and/or a human, and are *appended* here, never overwritten: re-judging with another model or
overriding by hand keeps the earlier opinions so judges can be compared. ``benchmark_cases.verdict`` is the
effective verdict derived from these rows by :func:`effective_judgement` (human > AI > exact > scripted).
"""
from __future__ import annotations

import time
from typing import Any

from finetune_studio.db.connection import cursor, new_id

VERDICTS = ("pass", "partial", "fail")
KINDS = ("human", "ai", "exact", "scripted")


def effective_judgement(rows: list[dict[str, Any]]) -> dict[str, Any] | None:
    """The judgement that decides a case. ``rows`` are in chronological order.

    A person beats a model, a model beats a matcher; within a kind the newest verdict wins. A person's newest row
    with an empty verdict is a *retraction*: no human opinion remains and the next kind decides. A model row with
    an empty verdict is just a failed call and hides nothing.
    """
    human = [r for r in rows if r.get("kind") == "human"]
    if human and human[-1].get("verdict") in VERDICTS:
        return human[-1]
    for kind in ("ai", "exact", "scripted"):
        valid = [r for r in rows if r.get("kind") == kind and r.get("verdict") in VERDICTS]
        if valid:
            return valid[-1]
    return None


def list_judgements(*, case_id: str | None = None, benchmark_id: str | None = None) -> list[dict[str, Any]]:
    """Judgement rows oldest first, for one case or a whole benchmark."""
    if not case_id and not benchmark_id:
        raise ValueError("case_id or benchmark_id required")
    col, val = ("case_id", case_id) if case_id else ("benchmark_id", benchmark_id)
    with cursor() as c:
        rows = c.execute(
            f"SELECT * FROM case_judgements WHERE {col} = ? ORDER BY created_at, rowid", (val,),
        ).fetchall()
    return [dict(r) for r in rows]


def add_judgement(
    case_id: str,
    *,
    kind: str,
    verdict: str,
    reasoning: str = "",
    confidence: float | None = None,
    provider_id: str = "",
    judge_model: str = "",
    error: str = "",
    prompt_version: str = "",
    apply: bool = True,
) -> dict[str, Any]:
    """Append a judgement and (by default) refresh the case's effective verdict. Returns the stored row."""
    if kind not in KINDS:
        raise ValueError(f"kind must be one of {KINDS}")
    if verdict and verdict not in VERDICTS:
        raise ValueError(f"verdict must be one of {VERDICTS} or empty")
    jid = new_id()
    now = time.time()
    with cursor() as c:
        case = c.execute("SELECT benchmark_id FROM benchmark_cases WHERE id = ?", (case_id,)).fetchone()
        if case is None:
            raise LookupError(f"case not found: {case_id}")
        c.execute(
            "INSERT INTO case_judgements (id, case_id, benchmark_id, kind, provider_id, judge_model, verdict, "
            "reasoning, confidence, error, prompt_version, created_at) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (jid, case_id, case["benchmark_id"], kind, provider_id, judge_model, verdict, reasoning, confidence, error,
             prompt_version, now),
        )
    if apply:
        apply_effective(case_id)
    return {"id": jid, "case_id": case_id, "kind": kind, "verdict": verdict, "reasoning": reasoning,
            "confidence": confidence, "provider_id": provider_id, "judge_model": judge_model, "error": error,
            "prompt_version": prompt_version, "created_at": now}


def apply_effective(case_id: str) -> dict[str, Any] | None:
    """Write the deciding judgement onto the case row (or clear it when none decides). Returns the winner."""
    winner = effective_judgement(list_judgements(case_id=case_id))
    with cursor() as c:
        if winner is None:
            c.execute(
                "UPDATE benchmark_cases SET verdict = '', judge = 'none', judge_model = '', judge_reasoning = '', "
                "scored_at = NULL WHERE id = ?", (case_id,),
            )
        else:
            c.execute(
                "UPDATE benchmark_cases SET verdict = ?, judge = ?, judge_model = ?, judge_reasoning = ?, "
                "scored_at = ? WHERE id = ?",
                (winner["verdict"], winner["kind"], winner.get("judge_model") or "",
                 winner.get("reasoning") or "", winner.get("created_at"), case_id),
            )
    return winner


def judged_case_ids(benchmark_id: str, *, kind: str, provider_id: str = "") -> set[str]:
    """Cases that already hold a verdict of ``kind`` (optionally from one provider) — for 'judge only the rest'."""
    sql = "SELECT DISTINCT case_id FROM case_judgements WHERE benchmark_id = ? AND kind = ? AND verdict != ''"
    args: list[str] = [benchmark_id, kind]
    if provider_id:
        sql += " AND provider_id = ?"
        args.append(provider_id)
    with cursor() as c:
        return {r["case_id"] for r in c.execute(sql, args).fetchall()}


def agreement(benchmark_id: str) -> dict[str, Any]:
    """How often each AI judge's verdict matches the human verdict, over cases a human has reviewed."""
    by_case: dict[str, dict[str, list[dict[str, Any]]]] = {}
    for row in list_judgements(benchmark_id=benchmark_id):
        by_case.setdefault(row["case_id"], {}).setdefault(row["kind"], []).append(row)
    stats: dict[str, dict[str, Any]] = {}
    reviewed = 0
    for kinds in by_case.values():
        human = effective_judgement(kinds.get("human", []))
        if human is None:
            continue
        reviewed += 1
        latest_per_model: dict[str, dict[str, Any]] = {}
        for row in kinds.get("ai", []):
            if row["verdict"] in VERDICTS:
                latest_per_model[row.get("judge_model") or row.get("provider_id") or "ai"] = row
        for model, row in latest_per_model.items():
            s = stats.setdefault(model, {"compared": 0, "agree": 0})
            s["compared"] += 1
            s["agree"] += int(row["verdict"] == human["verdict"])
    for s in stats.values():
        s["agreement"] = round(s["agree"] / s["compared"], 4) if s["compared"] else None
    return {"human_reviewed": reviewed, "judges": stats}
