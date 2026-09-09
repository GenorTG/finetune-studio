"""CRUD for `project_datasets` — per-project jsonl training data files.

Each dataset is a jsonl file on disk registered to a project. Datasets can come
from data-prep exports (`source='data-prep-export'`) or direct uploads
(`source='upload'`). The training tab reads from this table instead of asking
the user for a raw filesystem path.
"""
from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

from finetune_studio.config import settings
from finetune_studio.db.connection import cursor, new_id, row_to_dict


def datasets_dir(pid: str) -> Path:
    """Per-project datasets directory. Created on demand.

    Storage layout: <db_dir>/projects/<pid>/datasets/<name>.jsonl
    Using the DB's parent dir keeps data co-located with the rest of the
    studio's state, so a single backup covers everything.
    """
    base = Path(settings.db_path).parent / "projects" / pid / "datasets"
    base.mkdir(parents=True, exist_ok=True)
    return base


def create_dataset(project_id: str, name: str, data_path: str,
                   source: str = "upload", qa_count: int = 0,
                   size_bytes: int = 0) -> dict:
    """Register a jsonl file as a dataset for a project."""
    did = new_id()
    now = time.time()
    try:
        size_bytes = size_bytes or Path(data_path).stat().st_size
    except Exception:  # noqa: BLE001
        size_bytes = size_bytes or 0
    with cursor() as c:
        c.execute(
            "INSERT INTO project_datasets "
            "(id, project_id, name, data_path, source, qa_count, size_bytes, created_at) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
            (did, project_id, name, data_path, source, qa_count, size_bytes, now),
        )
    return get_dataset(did)  # type: ignore[return-value]


def get_dataset(did: str) -> dict | None:
    with cursor() as c:
        r = c.execute("SELECT * FROM project_datasets WHERE id = ?", (did,)).fetchone()
    return row_to_dict(r)


def get_dataset_by_path(pid: str, data_path: str) -> dict | None:
    """Look up by absolute path; used to dedupe registrations."""
    with cursor() as c:
        r = c.execute(
            "SELECT * FROM project_datasets WHERE project_id = ? AND data_path = ?",
            (pid, data_path),
        ).fetchone()
    return row_to_dict(r)


def list_datasets(project_id: str) -> list[dict]:
    with cursor() as c:
        rows = c.execute(
            "SELECT * FROM project_datasets WHERE project_id = ? ORDER BY created_at DESC",
            (project_id,),
        ).fetchall()
    return [row_to_dict(r) for r in rows]


def update_dataset(did: str, **fields: Any) -> dict | None:
    """Update qa_count/size_bytes last-modified time etc."""
    allowed = {"name", "qa_count", "size_bytes", "last_used_at"}
    sets, vals = [], []
    for k, v in fields.items():
        if k in allowed:
            sets.append(f"{k} = ?")
            vals.append(v)
    if not sets:
        return get_dataset(did)
    sets.append("last_used_at = ?")
    vals.append(time.time())
    vals.append(did)
    with cursor() as c:
        c.execute(f"UPDATE project_datasets SET {', '.join(sets)} WHERE id = ?", vals)
    return get_dataset(did)


def delete_dataset(did: str, remove_file: bool = False) -> dict:
    """Delete a dataset. If remove_file=True, also delete the underlying jsonl file."""
    ds = get_dataset(did)
    if not ds:
        return {"ok": False, "error": "not found"}
    if remove_file:
        try:
            Path(ds["data_path"]).unlink(missing_ok=True)
        except Exception as e:  # noqa: BLE001
            return {"ok": False, "error": f"db deleted but file remove failed: {e}"}
    with cursor() as c:
        c.execute("DELETE FROM project_datasets WHERE id = ?", (did,))
    return {"ok": True}


def count_qa_pairs(jsonl_path: str) -> int:
    """Count non-empty lines in a jsonl file (best-effort)."""
    try:
        n = 0
        with open(jsonl_path, "rb") as f:
            for line in f:
                if line.strip():
                    n += 1
        return n
    except Exception:  # noqa: BLE001
        return 0
