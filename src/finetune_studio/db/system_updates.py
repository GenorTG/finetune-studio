"""CRUD for `system_updates` — durable record of self-healing updates.

One row per 'POST /api/system/update' attempt. The log_text field
captures the full stdout/stderr of the update script so the UI can
show what happened during the install.
"""
from __future__ import annotations

import json
import time
from typing import Any

from finetune_studio.db.connection import cursor, new_id, row_to_dict


def _get(uid: str) -> dict | None:
    with cursor() as c:
        r = c.execute("SELECT * FROM system_updates WHERE id = ?", (uid,)).fetchone()
    return row_to_dict(r)


def create_update(*, mode: str = "update", options: dict | None = None,
                  triggered_by: str = "user") -> dict:
    uid = new_id()
    now = time.time()
    with cursor() as c:
        c.execute(
            "INSERT INTO system_updates "
            "(id, mode, status, options_json, triggered_by, created_at) "
            "VALUES (?, ?, 'queued', ?, ?, ?)",
            (uid, mode, json.dumps(options or {}), triggered_by, now),
        )
    return _get(uid)  # type: ignore[return-value]


def mark_running(uid: str) -> dict | None:
    return update_update(uid, status="running", started_at=time.time())


def mark_done(uid: str) -> dict | None:
    """Mark the update as done. Does NOT touch log_text — use
    append_log() to accumulate output as the script runs."""
    with cursor() as c:
        row = c.execute("SELECT started_at FROM system_updates WHERE id = ?", (uid,)).fetchone()
    started = row["started_at"] if row else None
    finished = time.time()
    duration_ms = int((finished - started) * 1000) if started else None
    return update_update(uid, status="done", finished_at=finished,
                         duration_ms=duration_ms)


def mark_failed(uid: str, *, error: str) -> dict | None:
    """Mark the update as failed. Does NOT touch log_text."""
    return update_update(uid, status="error", finished_at=time.time(),
                         error=error[:2000])


def mark_cancelled(uid: str) -> dict | None:
    return update_update(uid, status="cancelled", finished_at=time.time())


def append_log(uid: str, chunk: str, max_bytes: int = 256_000) -> dict | None:
    """Append a chunk to log_text. Truncates from the top when it grows
    past max_bytes so the DB row doesn't balloon on huge logs."""
    if not chunk:
        return _get(uid)
    with cursor() as c:
        row = c.execute("SELECT log_text FROM system_updates WHERE id = ?", (uid,)).fetchone()
        existing = row["log_text"] if row else ""
        combined = existing + chunk
        if len(combined) > max_bytes:
            # Trim from the top, keep the most recent max_bytes.
            combined = combined[-max_bytes:]
        c.execute("UPDATE system_updates SET log_text = ? WHERE id = ?",
                  (combined, uid))
    return _get(uid)


def update_update(uid: str, **fields: Any) -> dict | None:
    allowed = {
        "mode", "status",
        "started_at", "finished_at", "duration_ms",
        "log_text", "error", "options_json", "triggered_by",
    }
    sets, vals = [], []
    for k, v in fields.items():
        if k in allowed:
            if k == "options_json" and isinstance(v, (dict, list)):
                v = json.dumps(v)
            sets.append(f"{k} = ?")
            vals.append(v)
    if not sets:
        return _get(uid)
    vals.append(uid)
    with cursor() as c:
        c.execute(f"UPDATE system_updates SET {', '.join(sets)} WHERE id = ?", vals)
    return _get(uid)


def get_update(uid: str) -> dict | None:
    return _get(uid)


def list_recent(limit: int = 50) -> list[dict]:
    with cursor() as c:
        rows = c.execute(
            "SELECT * FROM system_updates ORDER BY created_at DESC LIMIT ?",
            (limit,),
        ).fetchall()
    return [row_to_dict(r) for r in rows]


def latest_in_progress() -> dict | None:
    """Most recent queued or running update — what the UI polls for status."""
    with cursor() as c:
        row = c.execute(
            "SELECT * FROM system_updates WHERE status IN ('queued', 'running') "
            "ORDER BY created_at DESC LIMIT 1",
        ).fetchone()
    return row_to_dict(row)
