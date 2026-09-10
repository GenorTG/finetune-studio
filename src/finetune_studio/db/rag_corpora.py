"""CRUD for `rag_corpora` — durable record of RAG build / ingest jobs.

One row per build attempt. Tracks timing, status, and final doc/chunk
counts. The `project_rags` row holds the latest summary; this table is
the history.
"""
from __future__ import annotations

import time
from typing import Any

from finetune_studio.db.connection import cursor, new_id, row_to_dict


def _get(rid: str) -> dict | None:
    with cursor() as c:
        r = c.execute("SELECT * FROM rag_corpora WHERE id = ?", (rid,)).fetchone()
    return row_to_dict(r)


def create_build(project_id: str, rag_id: str) -> dict:
    rid = new_id()
    now = time.time()
    with cursor() as c:
        c.execute(
            "INSERT INTO rag_corpora "
            "(id, project_id, rag_id, status, created_at) "
            "VALUES (?, ?, ?, 'queued', ?)",
            (rid, project_id, rag_id, now),
        )
    return _get(rid)  # type: ignore[return-value]


def mark_running(rid: str) -> dict | None:
    return update_build(rid, status="running", started_at=time.time())


def mark_done(rid: str, *, doc_count: int = 0, chunk_count: int = 0) -> dict | None:
    with cursor() as c:
        row = c.execute("SELECT started_at FROM rag_corpora WHERE id = ?", (rid,)).fetchone()
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
        c.execute(f"UPDATE rag_corpora SET {', '.join(sets)} WHERE id = ?", vals)
    return _get(rid)


def get_build(rid: str) -> dict | None:
    return _get(rid)


def list_for_rag(rag_id: str, limit: int = 50) -> list[dict]:
    with cursor() as c:
        rows = c.execute(
            "SELECT * FROM rag_corpora WHERE rag_id = ? "
            "ORDER BY created_at DESC LIMIT ?",
            (rag_id, limit),
        ).fetchall()
    return [row_to_dict(r) for r in rows]


def latest_for_rag(rag_id: str) -> dict | None:
    """The most recent build attempt for a RAG — what the UI shows as 'last build'."""
    with cursor() as c:
        row = c.execute(
            "SELECT * FROM rag_corpora WHERE rag_id = ? "
            "ORDER BY created_at DESC LIMIT 1",
            (rag_id,),
        ).fetchone()
    return row_to_dict(row)
