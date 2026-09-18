"""CRUD for `model_exports` \u2014 durable record of model exports.

One row per export attempt (currently only GGUF, but the table is
extensible). Tracks timing, status, output path, file size, and the
intermediate fp16 file when a quantized output went through two
stages (HF -> fp16 GGUF -> quantized GGUF).
"""
from __future__ import annotations

import time
from typing import Any

from finetune_studio.db.connection import cursor, new_id, row_to_dict


def _get(eid: str) -> dict | None:
    with cursor() as c:
        r = c.execute("SELECT * FROM model_exports WHERE id = ?", (eid,)).fetchone()
    return row_to_dict(r)


def create_export(project_id: str, run_id: str, *,
                  format: str = "gguf", quant: str = "Q4_K_M") -> dict:
    eid = new_id()
    now = time.time()
    with cursor() as c:
        c.execute(
            "INSERT INTO model_exports "
            "(id, project_id, run_id, format, quant, status, created_at) "
            "VALUES (?, ?, ?, ?, ?, 'queued', ?)",
            (eid, project_id, run_id, format, quant, now),
        )
    return _get(eid)  # type: ignore[return-value]


def mark_running(eid: str) -> dict | None:
    return update_export(eid, status="running", started_at=time.time())


def mark_done(eid: str, *, output_path: str = "", size_bytes: int = 0,
              size_human: str = "", intermediate_path: str = "") -> dict | None:
    with cursor() as c:
        row = c.execute("SELECT started_at FROM model_exports WHERE id = ?", (eid,)).fetchone()
    started = row["started_at"] if row else None
    finished = time.time()
    duration_ms = int((finished - started) * 1000) if started else None
    return update_export(eid, status="done", finished_at=finished,
                         duration_ms=duration_ms, output_path=output_path,
                         size_bytes=size_bytes, size_human=size_human,
                         intermediate_path=intermediate_path)


def mark_failed(eid: str, error: str) -> dict | None:
    return update_export(eid, status="error", finished_at=time.time(),
                        error=error[:1000])


def update_export(eid: str, **fields: Any) -> dict | None:
    allowed = {
        "format", "quant", "status",
        "started_at", "finished_at", "duration_ms",
        "output_path", "size_bytes", "size_human", "intermediate_path",
        "error",
    }
    sets, vals = [], []
    for k, v in fields.items():
        if k in allowed:
            sets.append(f"{k} = ?")
            vals.append(v)
    if not sets:
        return _get(eid)
    vals.append(eid)
    with cursor() as c:
        c.execute(f"UPDATE model_exports SET {', '.join(sets)} WHERE id = ?", vals)
    return _get(eid)


def get_export(eid: str) -> dict | None:
    return _get(eid)


def list_for_run(run_id: str, limit: int = 50) -> list[dict]:
    with cursor() as c:
        rows = c.execute(
            "SELECT * FROM model_exports WHERE run_id = ? "
            "ORDER BY created_at DESC LIMIT ?",
            (run_id, limit),
        ).fetchall()
    return [row_to_dict(r) for r in rows]


def list_for_project(project_id: str, limit: int = 100) -> list[dict]:
    with cursor() as c:
        rows = c.execute(
            "SELECT * FROM model_exports WHERE project_id = ? "
            "ORDER BY created_at DESC LIMIT ?",
            (project_id, limit),
        ).fetchall()
    return [row_to_dict(r) for r in rows]


def list_recent(limit: int = 50) -> list[dict]:
    with cursor() as c:
        rows = c.execute(
            "SELECT * FROM model_exports ORDER BY created_at DESC LIMIT ?",
            (limit,),
        ).fetchall()
    return [row_to_dict(r) for r in rows]


def reconcile_stale(error: str = "interrupted by service restart") -> int:
    """Mark in-flight model_exports as failed after a process restart.

    Returns the number of rows updated.
    """
    stale = ("queued", "running")
    placeholders = ", ".join("?" for _ in stale)
    now = time.time()
    with cursor() as c:
        rows = c.execute(
            f"SELECT id FROM model_exports WHERE status IN ({placeholders})",
            stale,
        ).fetchall()
        for r in rows:
            c.execute(
                "UPDATE model_exports SET status = 'failed', error = ?, finished_at = ? WHERE id = ?",
                (error, now, r["id"]),
            )
    return len(rows)
