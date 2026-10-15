"""CRUD for `brain_indexes` — the project's Smart Brain index row (the chat and the page attach to it)."""
from __future__ import annotations

import os
import time
from typing import Any

from finetune_studio.db.connection import cursor, new_id, row_to_dict


def _get(rid: str) -> dict | None:
    with cursor() as c:
        r = c.execute("SELECT * FROM brain_indexes WHERE id = ?", (rid,)).fetchone()
    return row_to_dict(r)


def create_brain_index(project_id: str, name: str, description: str = "",
               tags: str = "", store_path: str = "") -> dict:
    rid = new_id()
    now = time.time()
    if not store_path:
        from finetune_studio.data.fs.paths import brain_index_dir

        store_path = str(brain_index_dir(project_id))
    with cursor() as c:
        c.execute(
            "INSERT INTO brain_indexes (id, project_id, name, description, tags, store_path, created_at) "
            "VALUES (?, ?, ?, ?, ?, ?, ?)",
            (rid, project_id, name, description, tags, store_path, now),
        )
    return _get(rid)  # type: ignore[return-value]


def get_brain_index(rid: str) -> dict | None:
    return _get(rid)


def list_brain_indexes(project_id: str) -> list[dict]:
    with cursor() as c:
        rows = c.execute(
            "SELECT * FROM brain_indexes WHERE project_id = ? ORDER BY created_at DESC",
            (project_id,),
        ).fetchall()
    return [row_to_dict(r) for r in rows]


def update_brain_index(rid: str, **fields: Any) -> dict | None:
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
        c.execute(f"UPDATE brain_indexes SET {', '.join(sets)} WHERE id = ?", vals)
    return _get(rid)


def delete_brain_index(rid: str) -> bool:
    with cursor() as c:
        c.execute("DELETE FROM brain_indexes WHERE id = ?", (rid,))
    return True


def ensure_brain_index(
    project_id: str,
    store_path: str,
    *,
    name: str | None = None,
    doc_count: int = 0,
    chunk_count: int = 0,
) -> dict:
    """Create or update the ``brain_indexes`` row for a BrainIndex corpus path.

    Chat and the RAG page both key off ``brain_indexes.store_path``. The
    canonical BrainIndex build writes under ``brain/<pid>/``; this
    helper keeps a DB row pointing at that same directory so attachments
    stay in sync after every build/rebuild.
    """
    store_path = os.path.abspath(store_path)
    now = time.time()
    for index in list_brain_indexes(project_id):
        existing = os.path.abspath(str(index.get("store_path") or ""))
        if existing == store_path:
            updated = update_brain_index(
                index["id"],
                doc_count=int(doc_count),
                chunk_count=int(chunk_count),
                status="ready",
                last_build_at=now,
                last_build_status="ok",
                error="",
            )
            return updated or index
    display = name or f"{project_id} corpus"
    created = create_brain_index(
        project_id=project_id,
        name=display,
        description="BrainIndex corpus",
        store_path=store_path,
    )
    updated = update_brain_index(
        created["id"],
        doc_count=int(doc_count),
        chunk_count=int(chunk_count),
        status="ready",
        last_build_at=now,
        last_build_status="ok",
        error="",
    )
    return updated or created


def rebase_store_paths(old_root: str, new_root: str) -> int:
    """Point every index row under ``old_root`` at ``new_root`` (the index directory moved). Returns rows changed."""
    old_root, new_root = os.path.abspath(old_root), os.path.abspath(new_root)
    with cursor() as c:
        rows = c.execute("SELECT id, store_path FROM brain_indexes").fetchall()
        n = 0
        for r in rows:
            path = os.path.abspath(str(r["store_path"] or ""))
            if path == old_root or path.startswith(old_root + os.sep):
                c.execute("UPDATE brain_indexes SET store_path = ? WHERE id = ?",
                          (new_root + path[len(old_root):], r["id"]))
                n += 1
    return n
