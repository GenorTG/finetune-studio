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


def add_model_favorite(model_path: str, name: str = '', note: str = '') -> dict:
    fid = new_id()
    with cursor() as c:
        c.execute(
            "INSERT INTO model_favorites (id, model_path, name, note, added_at) VALUES (?, ?, ?, ?, ?)"
            " ON CONFLICT(model_path) DO UPDATE SET name=excluded.name, note=excluded.note",
            (fid, model_path, name, note, time.time()),
        )
    return {"id": fid, "path": model_path, "name": name, "note": note}


def remove_model_favorite(model_path: str) -> None:
    with cursor() as c:
        c.execute("DELETE FROM model_favorites WHERE model_path = ?", (model_path,))


def list_model_favorites() -> list[dict]:
    with cursor() as c:
        c.execute("SELECT id, model_path, name, note, added_at FROM model_favorites ORDER BY added_at DESC")
        return [{"id": r[0], "path": r[1], "name": r[2], "note": r[3], "added_at": r[4]} for r in c.fetchall()]


def is_model_favorited(model_path: str) -> bool:
    with cursor() as c:
        return c.execute("SELECT 1 FROM model_favorites WHERE model_path = ?", (model_path,)).fetchone() is not None
