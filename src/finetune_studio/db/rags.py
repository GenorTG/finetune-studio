"""CRUD for `project_rags` — vector stores scoped to a Project."""
from __future__ import annotations

import os
from typing import Any

from finetune_studio.db.connection import cursor, new_id, row_to_dict


def _get(rid: str) -> dict | None:
    with cursor() as c:
        r = c.execute("SELECT * FROM project_rags WHERE id = ?", (rid,)).fetchone()
    return row_to_dict(r)


def create_rag(project_id: str, name: str, description: str = "",
               tags: str = "", store_path: str = "") -> dict:
    rid = new_id()
    now = time_now()
    if not store_path:
        store_path = os.path.join("data", "rags", project_id, rid)
    with cursor() as c:
        c.execute(
            "INSERT INTO project_rags (id, project_id, name, description, tags, store_path, created_at) "
            "VALUES (?, ?, ?, ?, ?, ?, ?)",
            (rid, project_id, name, description, tags, store_path, now),
        )
    return _get(rid)  # type: ignore[return-value]


def get_rag(rid: str) -> dict | None:
    return _get(rid)


def list_rags(project_id: str) -> list[dict]:
    with cursor() as c:
        rows = c.execute(
            "SELECT * FROM project_rags WHERE project_id = ? ORDER BY created_at DESC",
            (project_id,),
        ).fetchall()
    return [row_to_dict(r) for r in rows]


def update_rag(rid: str, **fields: Any) -> dict | None:
    allowed = {
        "name", "description", "tags",
        "doc_count", "chunk_count",
        "status", "last_build_at", "last_build_status", "error",
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
        c.execute(f"UPDATE project_rags SET {', '.join(sets)} WHERE id = ?", vals)
    return _get(rid)


def delete_rag(rid: str) -> bool:
    with cursor() as c:
        c.execute("DELETE FROM project_rags WHERE id = ?", (rid,))
    return True


def time_now() -> float:
    import time as _t
    return _t.time()
