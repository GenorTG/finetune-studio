"""CRUD for `data_prep_runs` — durable record of data-prep jobs.

The in-memory `_RUNS` dict in `routes/data_prep.py` is still the source of
truth for in-progress polling, but anything that survives a restart (or
that the activity feed / dashboard reads) goes through this module.
"""
from __future__ import annotations

import json
import time
from typing import Any

from finetune_studio.db.connection import cursor, new_id, row_to_dict


def _get(rid: str) -> dict | None:
    with cursor() as c:
        r = c.execute("SELECT * FROM data_prep_runs WHERE id = ?", (rid,)).fetchone()
    return row_to_dict(r)


def create_run(project_id: str, filename: str = "", byte_count: int = 0,
               source_id: str = "", settings_obj: dict | None = None) -> dict:
    """Insert a fresh `queued` row. Caller sets started_at via mark_running."""
    rid = new_id()
    now = time.time()
    with cursor() as c:
        c.execute(
            "INSERT INTO data_prep_runs "
            "(id, project_id, source_id, filename, byte_count, status, "
            " settings_json, created_at) "
            "VALUES (?, ?, ?, ?, ?, 'queued', ?, ?)",
            (rid, project_id, source_id, filename, byte_count,
             json.dumps(settings_obj or {}), now),
        )
    return _get(rid)  # type: ignore[return-value]


def mark_running(rid: str) -> dict | None:
    return update_run(rid, status="running", started_at=time.time())


def mark_done(rid: str, *, qa_total: int = 0, qa_approved: int = 0,
              output_path: str = "") -> dict | None:
    """Mark the run as `done`, computing duration_ms from started_at."""
    with cursor() as c:
        row = c.execute("SELECT started_at FROM data_prep_runs WHERE id = ?", (rid,)).fetchone()
    started = row["started_at"] if row else None
    finished = time.time()
    duration_ms = int((finished - started) * 1000) if started else None
    return update_run(rid, status="done", finished_at=finished,
                      duration_ms=duration_ms, qa_total=qa_total,
                      qa_approved=qa_approved, output_path=output_path)


def mark_failed(rid: str, error: str) -> dict | None:
    return update_run(rid, status="error", finished_at=time.time(),
                      error=error[:500])


def update_run(rid: str, **fields: Any) -> dict | None:
    allowed = {
        "source_id", "filename", "byte_count", "status",
        "started_at", "finished_at", "duration_ms",
        "qa_total", "qa_approved", "output_path", "error",
        "settings_json",
    }
    sets, vals = [], []
    for k, v in fields.items():
        if k in allowed:
            if k == "settings_json" and isinstance(v, (dict, list)):
                v = json.dumps(v)
            sets.append(f"{k} = ?")
            vals.append(v)
    if not sets:
        return _get(rid)
    vals.append(rid)
    with cursor() as c:
        c.execute(f"UPDATE data_prep_runs SET {', '.join(sets)} WHERE id = ?", vals)
    return _get(rid)


def get_run(rid: str) -> dict | None:
    return _get(rid)


def list_for_project(project_id: str, limit: int = 100) -> list[dict]:
    """Return recent runs for the activity feed / project dashboard."""
    with cursor() as c:
        rows = c.execute(
            "SELECT * FROM data_prep_runs WHERE project_id = ? "
            "ORDER BY created_at DESC LIMIT ?",
            (project_id, limit),
        ).fetchall()
    return [row_to_dict(r) for r in rows]


def list_recent(limit: int = 50) -> list[dict]:
    """Across all projects — for the global activity feed."""
    with cursor() as c:
        rows = c.execute(
            "SELECT * FROM data_prep_runs ORDER BY created_at DESC LIMIT ?",
            (limit,),
        ).fetchall()
    return [row_to_dict(r) for r in rows]


def reconcile_stale(error: str = "interrupted by service restart") -> int:
    """Mark in-flight data_prep_runs as failed after a process restart.

    Returns the number of rows updated.
    """
    import time
    stale = ("queued", "running")
    placeholders = ", ".join("?" for _ in stale)
    now = time.time()
    with cursor() as c:
        rows = c.execute(
            f"SELECT id FROM data_prep_runs WHERE status IN ({placeholders})",
            stale,
        ).fetchall()
        for r in rows:
            c.execute(
                "UPDATE data_prep_runs SET status = 'failed', error = ?, finished_at = ? WHERE id = ?",
                (error, now, r["id"]),
            )
    return len(rows)
