"""CRUD for `benchmark_runs` + `benchmark_cases`."""
from __future__ import annotations

import json
import time

from finetune_studio.db.connection import cursor, new_id, row_to_dict

# ── benchmark_runs (parent) ───────────────────────────────────────────────

def _get(bid: str) -> dict | None:
    with cursor() as c:
        r = c.execute("SELECT * FROM benchmark_runs WHERE id = ?", (bid,)).fetchone()
    return row_to_dict(r)


def create_benchmark(run_id: str, suite_name: str, scores: dict,
                     time_ms: int = 0, cases: list[dict] | None = None) -> dict:
    """Create a benchmark run + (optionally) its per-case results."""
    bid = new_id()
    now = time.time()
    with cursor() as c:
        c.execute(
            "INSERT INTO benchmark_runs (id, run_id, suite_name, scores_json, time_ms, ran_at) "
            "VALUES (?, ?, ?, ?, ?, ?)",
            (bid, run_id, suite_name, json.dumps(scores), time_ms, now),
        )
        # Optional per-case rows (new in v2 schema — AI/human judging)
        if cases:
            for case in cases:
                cid = new_id()
                c.execute(
                    "INSERT INTO benchmark_cases (id, benchmark_id, run_id, case_name, category, "
                    "question, correct_answer, model_answer, transcript, judge, judge_model, "
                    "verdict, judge_reasoning, scored_at, scoring_method, validity, error, "
                    "judge_input, source_id, chunk_idx) "
                    "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                    (
                        cid, bid, run_id,
                        case.get("name", ""),
                        case.get("category", "general"),
                        case.get("question", ""),
                        case.get("correct_answer", ""),
                        case.get("model_answer", ""),
                        json.dumps(case.get("transcript", [])),
                        case.get("judge", "none"),
                        case.get("judge_model", ""),
                        case.get("verdict", ""),
                        case.get("judge_reasoning", ""),
                        case.get("scored_at"),
                        case.get("scoring_method", ""),
                        case.get("validity", ""),
                        case.get("error", ""),
                        json.dumps(case.get("judge_input", {})),
                        case.get("source_id", ""),
                        int(case.get("chunk_idx") or 0),
                    ),
                )
    return _get(bid)  # type: ignore[return-value]


def get_benchmark(bid: str) -> dict | None:
    return _get(bid)


def list_benchmarks(run_id: str | None = None) -> list[dict]:
    with cursor() as c:
        if run_id:
            rows = c.execute(
                "SELECT * FROM benchmark_runs WHERE run_id = ? ORDER BY ran_at DESC",
                (run_id,),
            ).fetchall()
        else:
            rows = c.execute("SELECT * FROM benchmark_runs ORDER BY ran_at DESC").fetchall()
    return [row_to_dict(r) for r in rows]


# ── benchmark_cases (per-test results) ────────────────────────────────────

def create_case(benchmark_id: str, run_id: str, name: str, category: str,
                question: str, correct_answer: str, model_answer: str,
                transcript: list, judge: str = "none", judge_model: str = "",
                verdict: str = "", judge_reasoning: str = "",
                scored_at: float | None = None, scoring_method: str = "",
                validity: str = "", error: str = "", judge_input: dict | None = None,
                source_id: str = "", chunk_idx: int = 0) -> str:
    """Insert a single benchmark case row. Returns the new id."""
    cid = new_id()
    with cursor() as c:
        c.execute(
            "INSERT INTO benchmark_cases (id, benchmark_id, run_id, case_name, category, "
            "question, correct_answer, model_answer, transcript, judge, judge_model, "
            "verdict, judge_reasoning, scored_at) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (
                cid, benchmark_id, run_id, name, category,
                question, correct_answer, model_answer,
                json.dumps(transcript), judge, judge_model,
                verdict, judge_reasoning, scored_at,
            ),
        )
    return cid


def list_cases(benchmark_id: str) -> list[dict]:
    """Return all cases for a benchmark run."""
    with cursor() as c:
        rows = c.execute(
            "SELECT * FROM benchmark_cases WHERE benchmark_id = ? ORDER BY rowid",
            (benchmark_id,),
        ).fetchall()
    out = []
    for r in rows:
        d = dict(r)
        if "transcript" in d and isinstance(d["transcript"], str) and d["transcript"]:
            try:
                d["transcript"] = json.loads(d["transcript"])
            except (TypeError, json.JSONDecodeError):
                d["transcript"] = []
        if "judge_input" in d and isinstance(d["judge_input"], str) and d["judge_input"]:
            try:
                d["judge_input"] = json.loads(d["judge_input"])
            except (TypeError, json.JSONDecodeError):
                d["judge_input"] = {}
        out.append(d)
    return out


def update_case(cid: str, **kwargs) -> None:
    """Partial update of a case row (used by judge)."""
    if not kwargs:
        return
    # JSON-encode transcript if present
    if "transcript" in kwargs and not isinstance(kwargs["transcript"], str):
        kwargs["transcript"] = json.dumps(kwargs["transcript"])
    sets = ", ".join(f"{k} = ?" for k in kwargs)
    vals = list(kwargs.values()) + [cid]
    with cursor() as c:
        c.execute(f"UPDATE benchmark_cases SET {sets} WHERE id = ?", vals)


def update_benchmark_scores(bid: str, scores: dict) -> None:
    """Update the scores_json for a benchmark run (used after judging)."""
    with cursor() as c:
        c.execute(
            "UPDATE benchmark_runs SET scores_json = ? WHERE id = ?",
            (json.dumps(scores), bid),
        )
