"""CRUD for the `projects` table."""
from __future__ import annotations

import time
from typing import Any

from finetune_studio.db.connection import cursor, new_id, row_to_dict


def _get(pid: str) -> dict | None:
    with cursor() as c:
        r = c.execute("SELECT * FROM projects WHERE id = ?", (pid,)).fetchone()
    return row_to_dict(r)


def create_project(name: str, description: str = "", base_model: str = "",
                   system_prompt: str = "") -> dict:
    pid = new_id()
    now = time.time()
    with cursor() as c:
        c.execute(
            "INSERT INTO projects (id, name, description, base_model, system_prompt, created_at, updated_at) "
            "VALUES (?, ?, ?, ?, ?, ?, ?)",
            (pid, name, description, base_model, system_prompt, now, now),
        )
    return _get(pid)  # type: ignore[return-value]


def get_project(pid: str) -> dict | None:
    return _get(pid)


def list_projects() -> list[dict]:
    with cursor() as c:
        rows = c.execute("SELECT * FROM projects ORDER BY updated_at DESC").fetchall()
    return [row_to_dict(r) for r in rows]


def update_project(pid: str, **fields: Any) -> dict | None:
    allowed = {"name", "description", "base_model", "system_prompt", "production_run", "tags", "notes"}
    sets, vals = [], []
    for k, v in fields.items():
        if k in allowed:
            sets.append(f"{k} = ?")
            vals.append(v)
    if not sets:
        return _get(pid)
    sets.append("updated_at = ?")
    vals.append(time.time())
    vals.append(pid)
    with cursor() as c:
        c.execute(f"UPDATE projects SET {', '.join(sets)} WHERE id = ?", vals)
    return _get(pid)


def delete_project(pid: str) -> bool:
    with cursor() as c:
        c.execute("DELETE FROM projects WHERE id = ?", (pid,))
    return True
