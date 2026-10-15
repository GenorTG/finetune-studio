"""CRUD for `brain_builds` — durable record of RAG build / ingest jobs.

One row per build attempt. Tracks timing, status, and final doc/chunk
counts. The `brain_indexes` row holds the latest summary; this table is
the history.
"""
from __future__ import annotations

import time
from typing import Any

from finetune_studio.db.connection import cursor, new_id, row_to_dict


def _get(rid: str) -> dict | None:
    with cursor() as c:
        r = c.execute("SELECT * FROM brain_builds WHERE id = ?", (rid,)).fetchone()
    return row_to_dict(r)


def create_build(project_id: str, index_id: str) -> dict:
    rid = new_id()
    now = time.time()
    with cursor() as c:
        c.execute(
            "INSERT INTO brain_builds "
            "(id, project_id, index_id, status, created_at) "
            "VALUES (?, ?, ?, 'queued', ?)",
            (rid, project_id, index_id, now),
        )
    return _get(rid)  # type: ignore[return-value]


def mark_running(rid: str) -> dict | None:
    return update_build(rid, status="running", started_at=time.time())


def mark_done(rid: str, *, doc_count: int = 0, chunk_count: int = 0) -> dict | None:
    with cursor() as c:
        row = c.execute("SELECT started_at FROM brain_builds WHERE id = ?", (rid,)).fetchone()
    started = row["started_at"] if row else None
    finished = time.time()
    duration_ms = int((finished - started) * 1000) if started else None
    return update_build(rid, status="done", finished_at=finished,
                        duration_ms=duration_ms, doc_count=doc_count,
                        chunk_count=chunk_count)


def mark_failed(rid: str, error: str) -> dict | None:
    return update_build(rid, status="error", finished_at=time.time(),
                        error=error[:500])


def update_build(rid: str, **fields: Any) -> dict | None:
    allowed = {
        "status", "started_at", "finished_at", "duration_ms",
        "doc_count", "chunk_count", "error",
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
        c.execute(f"UPDATE brain_builds SET {', '.join(sets)} WHERE id = ?", vals)
    return _get(rid)


def get_build(rid: str) -> dict | None:
    return _get(rid)


def list_for_index(index_id: str, limit: int = 50) -> list[dict]:
    with cursor() as c:
        rows = c.execute(
            "SELECT * FROM brain_builds WHERE index_id = ? "
            "ORDER BY created_at DESC LIMIT ?",
            (index_id, limit),
        ).fetchall()
    return [row_to_dict(r) for r in rows]


def list_recent(limit: int = 50) -> list[dict]:
    """Across all indexes/projects — for the global activity feed."""
    with cursor() as c:
        rows = c.execute(
            "SELECT * FROM brain_builds ORDER BY created_at DESC LIMIT ?",
            (limit,),
        ).fetchall()
    return [row_to_dict(r) for r in rows]


def latest_for_index(index_id: str) -> dict | None:
    """The most recent build attempt for an index — what the UI shows as 'last build'."""
    with cursor() as c:
        row = c.execute(
            "SELECT * FROM brain_builds WHERE index_id = ? "
            "ORDER BY created_at DESC LIMIT 1",
            (index_id,),
        ).fetchone()
    return row_to_dict(row)


def latest_for_project(project_id: str) -> dict | None:
    """The project's most recent index build (a version pins it)."""
    with cursor() as c:
        row = c.execute("SELECT * FROM brain_builds WHERE project_id = ? ORDER BY created_at DESC LIMIT 1",
                        (project_id,)).fetchone()
    return row_to_dict(row)


def fail_if_active(rid: str, error: str) -> bool:
    """``mark_failed`` only while the row is queued/running, in one statement (the build job is another process: a
    read-then-write could land after it recorded its result). True when the row was updated."""
    with cursor() as c:
        cur = c.execute("UPDATE brain_builds SET status = 'error', error = ?, finished_at = ? "
                        "WHERE id = ? AND status IN ('queued', 'running')", (error[:500], time.time(), rid))
        return cur.rowcount > 0


def reconcile_stale(error: str = "interrupted by service restart",
                    keep: set[str] | frozenset[str] = frozenset()) -> int:
    """Mark in-flight brain_builds rows as failed after a web restart, except ``keep`` (their job process is alive).

    Returns the number of rows updated.
    """
    stale = ("queued", "running")
    placeholders = ", ".join("?" for _ in stale)
    now = time.time()
    with cursor() as c:
        rows = c.execute(
            f"SELECT id FROM brain_builds WHERE status IN ({placeholders})",
            stale,
        ).fetchall()
        rows = [r for r in rows if r["id"] not in keep]
        for r in rows:
            c.execute(
                "UPDATE brain_builds SET status = 'failed', error = ?, finished_at = ? WHERE id = ?",
                (error, now, r["id"]),
            )
    return len(rows)
