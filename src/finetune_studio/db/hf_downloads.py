"""CRUD for `hf_downloads` — durable record of HF Hub download jobs.

Replaces the in-memory `_DOWNLOADS` dict in `routes/hf_models.py` so a
download survives a service restart and the UI can show history.
"""
from __future__ import annotations

import time
from typing import Any

from finetune_studio.db.connection import cursor, new_id, row_to_dict


def _get(rid: str) -> dict | None:
    with cursor() as c:
        r = c.execute("SELECT * FROM hf_downloads WHERE id = ?", (rid,)).fetchone()
    return row_to_dict(r)


def create_job(repo_id: str, filename: str = "") -> dict:
    rid = new_id()
    now = time.time()
    with cursor() as c:
        c.execute(
            "INSERT INTO hf_downloads "
            "(id, repo_id, filename, status, created_at) "
            "VALUES (?, ?, ?, 'queued', ?)",
            (rid, repo_id, filename, now),
        )
    return _get(rid)  # type: ignore[return-value]


def mark_running(rid: str) -> dict | None:
    return update_job(rid, status="downloading", started_at=time.time())


def mark_done(rid: str, *, path: str = "", bytes_total: int | None = None,
              bytes_done: int | None = None) -> dict | None:
    with cursor() as c:
        row = c.execute("SELECT started_at FROM hf_downloads WHERE id = ?", (rid,)).fetchone()
    started = row["started_at"] if row else None
    finished = time.time()
    duration_ms = int((finished - started) * 1000) if started else None
    return update_job(rid, status="completed", finished_at=finished,
                      duration_ms=duration_ms, path=path,
                      bytes_total=bytes_total, bytes_done=bytes_done)


def mark_failed(rid: str, error: str) -> dict | None:
    return update_job(rid, status="error", finished_at=time.time(),
                      error=error[:500])


def mark_cancelled(rid: str) -> dict | None:
    return update_job(rid, status="cancelled", finished_at=time.time(),
                      error="user cancelled")


def update_job(rid: str, **fields: Any) -> dict | None:
    allowed = {
        "status", "started_at", "finished_at", "duration_ms",
        "bytes_total", "bytes_done", "path", "error",
    }
    sets, vals = [], []
    for k, v in fields.items():
        if k in allowed:
            sets.append(f"{k} = ?")
            vals.append(v)
    if not sets:
        return _get(rid)
    vals.append(rid)
    with cursor() as c:
        c.execute(f"UPDATE hf_downloads SET {', '.join(sets)} WHERE id = ?", vals)
    return _get(rid)


def get_job(rid: str) -> dict | None:
    return _get(rid)


def list_recent(limit: int = 50) -> list[dict]:
    """Across all repos — for the dashboard / activity feed."""
    with cursor() as c:
        rows = c.execute(
            "SELECT * FROM hf_downloads ORDER BY created_at DESC LIMIT ?",
            (limit,),
        ).fetchall()
    return [row_to_dict(r) for r in rows]


def list_in_progress(limit: int = 50) -> list[dict]:
    """All jobs that are queued or actively downloading — restored on startup
    so the UI can reattach to running jobs after a service restart."""
    with cursor() as c:
        rows = c.execute(
            "SELECT * FROM hf_downloads WHERE status IN ('queued', 'downloading') "
            "ORDER BY created_at DESC LIMIT ?",
            (limit,),
        ).fetchall()
    return [row_to_dict(r) for r in rows]
