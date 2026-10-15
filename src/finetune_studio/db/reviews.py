"""Row-level approve/reject/edit decisions for datasets (`data_review`)."""
from __future__ import annotations

import time

from finetune_studio.db.connection import cursor, new_id, row_to_dict


def record_review(project_id: str, dataset: str, row_index: int,
                  decision: str, edited_json: str | None = None) -> dict:
    """Record a row-level review decision. Overwrites prior decision for (project, dataset, row)."""
    rid = new_id()
    now = time.time()
    with cursor() as c:
        c.execute(
            "DELETE FROM data_review WHERE project_id = ? AND dataset = ? AND row_index = ?",
            (project_id, dataset, row_index),
        )
        c.execute(
            "INSERT INTO data_review (id, project_id, dataset, row_index, decision, edited_json, reviewed_at) "
            "VALUES (?, ?, ?, ?, ?, ?, ?)",
            (rid, project_id, dataset, row_index, decision, edited_json, now),
        )
    return {"id": rid, "row_index": row_index, "decision": decision}


def list_review(project_id: str, dataset: str) -> list[dict]:
    with cursor() as c:
        rows = c.execute(
            "SELECT * FROM data_review WHERE project_id = ? AND dataset = ?",
            (project_id, dataset),
        ).fetchall()
    return [row_to_dict(r) for r in rows]
