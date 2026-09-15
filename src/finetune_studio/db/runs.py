"""CRUD for `training_runs`."""
from __future__ import annotations

import json
import time
from typing import Any

from finetune_studio.db.connection import cursor, new_id, row_to_dict


def _get(rid: str) -> dict | None:
    with cursor() as c:
        r = c.execute("SELECT * FROM training_runs WHERE id = ?", (rid,)).fetchone()
    return row_to_dict(r)


def create_run(project_id: str, name: str, base_model: str = "",
               data_path: str = "", rag_ids: list | None = None,
               settings_obj: dict | None = None, system_prompt: str = "",
               system_prompt_mode: str = "bake",
               parent_run_id: str | None = None, notes: str = "") -> dict:
    rid = new_id()
    now = time.time()
    with cursor() as c:
        c.execute(
            "INSERT INTO training_runs (id, project_id, name, base_model, data_path, "
            "rag_ids_json, settings_json, system_prompt, system_prompt_mode, parent_run_id, notes, created_at) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (rid, project_id, name, base_model, data_path,
             json.dumps(rag_ids or []), json.dumps(settings_obj or {}),
             system_prompt, system_prompt_mode, parent_run_id, notes, now),
        )
    return _get(rid)  # type: ignore[return-value]


def get_run(rid: str) -> dict | None:
    return _get(rid)


def list_runs(project_id: str | None = None) -> list[dict]:
    with cursor() as c:
        if project_id:
            rows = c.execute(
                "SELECT * FROM training_runs WHERE project_id = ? ORDER BY created_at DESC",
                (project_id,),
            ).fetchall()
        else:
            rows = c.execute("SELECT * FROM training_runs ORDER BY created_at DESC").fetchall()
    runs = [row_to_dict(r) for r in rows]
    # Compute duration from started_at/finished_at if not set
    for run in runs:
        started = run.get("started_at")
        finished = run.get("finished_at")
        if started and finished:
            run["duration"] = finished - started
        else:
            run["duration"] = None
        # Extract final_loss from metrics_json if present
        metrics = run.get("metrics_json", {})
        if isinstance(metrics, str):
            import json as _json
            try:
                metrics = _json.loads(metrics)
            except Exception:
                metrics = {}
        run["final_loss"] = metrics.get("final_loss") if isinstance(metrics, dict) else None
    return runs


def update_run(rid: str, **fields: Any) -> dict | None:
    allowed = {
        "name", "base_model", "data_path", "system_prompt",
        "status", "started_at", "finished_at", "output_path",
        "metrics_json", "notes", "error", "parent_run_id", "final_loss",
    }
    sets, vals = [], []
    for k, v in fields.items():
        if k in allowed:
            if k in ("metrics_json",) and isinstance(v, (dict, list)):
                v = json.dumps(v)
            sets.append(f"{k} = ?")
            vals.append(v)
    if not sets:
        return _get(rid)
    vals.append(rid)
    with cursor() as c:
        c.execute(f"UPDATE training_runs SET {', '.join(sets)} WHERE id = ?", vals)
    return _get(rid)


def delete_run(rid: str) -> bool:
    with cursor() as c:
        c.execute("DELETE FROM training_runs WHERE id = ?", (rid,))
    return True


_STALE_RUN_STATUSES: tuple[str, ...] = (
    "queued", "loading", "training", "saving", "running",
)
_STALE_RUN_ERROR = "Interrupted: server restarted during this run"


def reconcile_stale_runs() -> int:
    """Mark in-flight training_runs as failed after a process restart.

    Any row still in queued/loading/training/saving/running cannot still be
    running after the server process died — mark them failed so the UI does
    not show forever-spinning orphans. Returns the number of rows updated.
    """
    placeholders = ", ".join("?" for _ in _STALE_RUN_STATUSES)
    now = time.time()
    with cursor() as c:
        rows = c.execute(
            f"SELECT id FROM training_runs WHERE status IN ({placeholders})",
            _STALE_RUN_STATUSES,
        ).fetchall()
        for r in rows:
            c.execute(
                "UPDATE training_runs SET status = ?, error = ?, finished_at = ? "
                "WHERE id = ?",
                ("failed", _STALE_RUN_ERROR, now, r["id"]),
            )
    return len(rows)
