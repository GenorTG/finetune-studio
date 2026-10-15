"""Durable log for application operations without a job-specific table."""
from __future__ import annotations

import time
from typing import Any

from finetune_studio.db.connection import cursor, new_id, row_to_dict


def record(
    *,
    kind: str,
    operation: str,
    method: str,
    path: str,
    project_id: str = "",
    status: str = "done",
    http_status: int = 200,
    message: str = "",
    started_at: float | None = None,
    finished_at: float | None = None,
) -> dict[str, Any]:
    """Persist one completed request operation and return its row."""
    started = started_at if started_at is not None else time.time()
    finished = finished_at if finished_at is not None else time.time()
    event_id = new_id()
    with cursor() as c:
        c.execute(
            "INSERT INTO activity_events "
            "(id, project_id, kind, operation, method, path, status, "
            "http_status, message, started_at, finished_at, created_at) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (event_id, project_id, kind, operation, method, path, status,
             http_status, message, started, finished, finished),
        )
    with cursor() as c:
        row = c.execute(
            "SELECT * FROM activity_events WHERE id = ?", (event_id,)
        ).fetchone()
    return row_to_dict(row) or {}


def list_recent(limit: int = 100) -> list[dict[str, Any]]:
    """Return newest operation events across all projects."""
    with cursor() as c:
        rows = c.execute(
            "SELECT * FROM activity_events ORDER BY created_at DESC LIMIT ?",
            (limit,),
        ).fetchall()
    return [row_to_dict(row) or {} for row in rows]
