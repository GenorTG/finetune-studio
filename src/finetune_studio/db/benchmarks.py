"""CRUD for `benchmark_runs` + `benchmark_cases`.

A benchmark run is a *test run*: it is created first (status ``running``), its cases are appended with the raw
transcript as the model answers them, and judging is a second step that only appends to `case_judgements`
(see :mod:`finetune_studio.db.judgements`). The run row also carries the progress of both jobs, so a page reload
or a second tab reads the same truth.
"""
from __future__ import annotations

import json
import time
from typing import Any

from finetune_studio.db.connection import cursor, new_id, row_to_dict

# ── benchmark_runs (parent) ───────────────────────────────────────────────

RUN_STATUSES = ("queued", "running", "done", "failed", "cancelled")
TERMINAL_RUN_STATUSES = frozenset({"done", "failed", "cancelled"})

_RUN_COLUMNS = frozenset({
    "suite_name", "model_path", "time_ms", "status", "kind", "scoring", "error", "progress_done",
    "progress_total", "heartbeat_at", "judge_status", "judge_provider_id", "judge_model", "judge_done",
    "judge_total", "judge_error",
})


def _get(bid: str) -> dict | None:
    with cursor() as c:
        r = c.execute("SELECT * FROM benchmark_runs WHERE id = ?", (bid,)).fetchone()
    return _decode_run(row_to_dict(r))


def _decode_run(run: dict | None) -> dict | None:
    """``config_json`` text -> ``config`` dict (row_to_dict does not know this column)."""
    if run is None:
        return None
    raw = run.pop("config_json", None)
    try:
        config = json.loads(raw) if isinstance(raw, str) and raw else {}
    except json.JSONDecodeError:
        config = {}
    run["config"] = config if isinstance(config, dict) else {}
    return run


def create_benchmark(run_id: str, suite_name: str, scores: dict,
                     time_ms: int = 0, cases: list[dict] | None = None,
                     model_path: str = "", *, status: str = "done", kind: str = "suite",
                     scoring: str = "judge", config: dict[str, Any] | None = None,
                     progress_total: int = 0) -> dict:
    """Create a benchmark run + (optionally) its per-case results.

    ``status="running"`` with no cases is how a live test run starts; use :func:`create_case` to append cases.
    """
    bid = new_id()
    now = time.time()
    n_cases = len(cases or [])
    with cursor() as c:
        c.execute(
            "INSERT INTO benchmark_runs (id, run_id, suite_name, model_path, scores_json, time_ms, ran_at, "
            "status, kind, scoring, config_json, progress_done, progress_total, heartbeat_at) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (bid, run_id, suite_name, model_path, json.dumps(scores), time_ms, now, status, kind, scoring,
             json.dumps(config or {}), n_cases, progress_total or n_cases, now),
        )
    for case in cases or []:
        create_case(
            bid, run_id, case.get("name", ""), case.get("category", "general"),
            case.get("question", ""), case.get("correct_answer", ""), case.get("model_answer", ""),
            case.get("transcript", []), judge=case.get("judge", "none"), judge_model=case.get("judge_model", ""),
            verdict=case.get("verdict", ""), judge_reasoning=case.get("judge_reasoning", ""),
            scored_at=case.get("scored_at"), scoring_method=case.get("scoring_method", ""),
            validity=case.get("validity", ""), error=case.get("error", ""), judge_input=case.get("judge_input"),
            source_id=case.get("source_id", ""), chunk_idx=case.get("chunk_idx", 0),
            keywords=case.get("keywords"), expect_abstain=bool(case.get("expect_abstain")),
            record_judgement=bool(case.get("verdict")), time_ms=float(case.get("time_ms") or 0),
        )
    return _get(bid)  # type: ignore[return-value]


def get_benchmark(bid: str) -> dict | None:
    return _get(bid)


def update_benchmark(bid: str, **fields: Any) -> None:
    """Partial update of a run row (status, progress, judge job state ...). Unknown columns are an error."""
    if not fields:
        return
    unknown = set(fields) - _RUN_COLUMNS
    if unknown:
        raise ValueError(f"unknown benchmark_runs columns: {sorted(unknown)}")
    sets = ", ".join(f"{k} = ?" for k in fields)
    with cursor() as c:
        c.execute(f"UPDATE benchmark_runs SET {sets} WHERE id = ?", [*fields.values(), bid])


def list_benchmarks(run_id: str | None = None) -> list[dict]:
    with cursor() as c:
        if run_id:
            rows = c.execute(
                "SELECT * FROM benchmark_runs WHERE run_id = ? ORDER BY ran_at DESC",
                (run_id,),
            ).fetchall()
        else:
            rows = c.execute("SELECT * FROM benchmark_runs ORDER BY ran_at DESC").fetchall()
    return [_decode_run(row_to_dict(r)) for r in rows]  # type: ignore[misc]


def list_for_project(pid: str, *, limit: int = 50) -> list[dict]:
    """Test runs of a project (newest first) with the owning training run's name attached."""
    with cursor() as c:
        rows = c.execute(
            "SELECT br.*, tr.name AS run_name FROM benchmark_runs br "
            "JOIN training_runs tr ON tr.id = br.run_id "
            "WHERE tr.project_id = ? ORDER BY br.ran_at DESC LIMIT ?",
            (pid, limit),
        ).fetchall()
    return [_decode_run(row_to_dict(r)) for r in rows]  # type: ignore[misc]


def list_unfinished() -> list[dict]:
    """Runs whose test job or judge job was left ``running``/``queued`` (the process died mid-job)."""
    with cursor() as c:
        rows = c.execute(
            "SELECT * FROM benchmark_runs WHERE status IN ('queued', 'running') OR judge_status = 'running'"
        ).fetchall()
    return [_decode_run(row_to_dict(r)) for r in rows]  # type: ignore[misc]


def reconcile_stale_benchmarks(cause: str = "") -> int:
    """Startup: a test run or judge job left running by a dead process becomes failed (its saved cases are kept)."""
    reason = "interrupted: the server restarted while this job was running" + cause
    with cursor() as c:
        n = c.execute(
            "UPDATE benchmark_runs SET status = 'failed', error = ? WHERE status IN ('queued', 'running')", (reason,),
        ).rowcount
        n += c.execute(
            "UPDATE benchmark_runs SET judge_status = 'failed', judge_error = ? WHERE judge_status = 'running'", (reason,),
        ).rowcount
    return n


def list_recent(limit: int = 50) -> list[dict]:
    """Across all runs — for the global activity feed."""
    with cursor() as c:
        rows = c.execute(
            "SELECT * FROM benchmark_runs ORDER BY ran_at DESC LIMIT ?",
            (limit,),
        ).fetchall()
    return [_decode_run(row_to_dict(r)) for r in rows]  # type: ignore[misc]


# ── benchmark_cases (per-test results) ────────────────────────────────────

def create_case(benchmark_id: str, run_id: str, name: str, category: str,
                question: str, correct_answer: str, model_answer: str,
                transcript: list, judge: str = "none", judge_model: str = "",
                verdict: str = "", judge_reasoning: str = "",
                scored_at: float | None = None, scoring_method: str = "",
                validity: str = "", error: str = "", judge_input: dict | None = None,
                source_id: str = "", chunk_idx: int = 0,
                keywords: list | None = None, expect_abstain: bool = False,
                record_judgement: bool = False, time_ms: float = 0.0) -> str:
    """Insert a single benchmark case row. Returns the new id.

    A test run stores the answer only (``verdict=""``). ``record_judgement`` is for rows that arrive already
    scored (exact-match public benchmarks): the verdict is mirrored into `case_judgements` so history is whole.
    """
    cid = new_id()
    with cursor() as c:
        c.execute(
            "INSERT INTO benchmark_cases (id, benchmark_id, run_id, case_name, category, "
            "question, correct_answer, model_answer, transcript, judge, judge_model, "
            "verdict, judge_reasoning, scored_at, scoring_method, validity, error, "
            "judge_input, source_id, chunk_idx, keywords, expect_abstain, time_ms) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (
                cid, benchmark_id, run_id, name, category,
                question, correct_answer, model_answer,
                json.dumps(transcript), judge, judge_model,
                verdict, judge_reasoning, scored_at,
                scoring_method, validity, error,
                json.dumps(judge_input or {}), source_id, int(chunk_idx or 0),
                json.dumps([str(k) for k in (keywords or [])]), int(bool(expect_abstain)), float(time_ms or 0),
            ),
        )
        if record_judgement and verdict:
            kind = {"human": "human", "ai": "ai", "local": "ai", "exact": "exact"}.get(judge, "scripted")
            c.execute(
                "INSERT INTO case_judgements (id, case_id, benchmark_id, kind, provider_id, judge_model, verdict, "
                "reasoning, confidence, error, created_at) VALUES (?, ?, ?, ?, '', ?, ?, ?, NULL, '', ?)",
                (new_id(), cid, benchmark_id, kind, judge_model, verdict, judge_reasoning, scored_at or time.time()),
            )
    return cid


def _decode_case(d: dict) -> dict:
    for col, empty in (("transcript", []), ("judge_input", {}), ("keywords", [])):
        if isinstance(d.get(col), str) and d[col]:
            try:
                d[col] = json.loads(d[col])
            except (TypeError, json.JSONDecodeError):
                d[col] = empty
        elif col in d and not d[col]:
            d[col] = empty
    d["expect_abstain"] = bool(d.get("expect_abstain"))
    return d


def list_cases(benchmark_id: str) -> list[dict]:
    """Return all cases for a benchmark run."""
    with cursor() as c:
        rows = c.execute(
            "SELECT * FROM benchmark_cases WHERE benchmark_id = ? ORDER BY rowid",
            (benchmark_id,),
        ).fetchall()
    return [_decode_case(dict(r)) for r in rows]


def get_case(cid: str) -> dict | None:
    with cursor() as c:
        r = c.execute("SELECT * FROM benchmark_cases WHERE id = ?", (cid,)).fetchone()
    return _decode_case(dict(r)) if r else None


_CASE_COLUMNS = frozenset({
    "case_name", "category", "question", "correct_answer", "model_answer",
    "transcript", "judge", "judge_model", "verdict", "judge_reasoning",
    "scored_at", "scoring_method", "validity", "error", "judge_input",
    "source_id", "chunk_idx", "keywords", "expect_abstain", "time_ms",
})


def update_case(cid: str, **kwargs) -> None:
    """Partial update of a case row."""
    if not kwargs:
        return
    unknown = set(kwargs) - _CASE_COLUMNS
    if unknown:
        raise ValueError(f"unknown benchmark_cases columns: {sorted(unknown)}")
    for key in ("transcript", "judge_input", "keywords"):
        if key in kwargs and not isinstance(kwargs[key], str):
            kwargs[key] = json.dumps(kwargs[key])
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
