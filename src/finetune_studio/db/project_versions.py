"""CRUD for `project_versions` — immutable, named snapshots of project state.

A version is a manifest that pins the exact inputs used to produce an output:
datasets, selected source files, RAG corpora, training runs, and the base
model. Versions are append-only; "building on" an old version creates a NEW
version whose manifest inherits the parent's pins (branching without
mutation). The runtime artifacts (jsonl files, corpus dirs, adapters) live on
disk and are referenced by id/path from the manifest — the manifest is the
index, not the payload.
"""
from __future__ import annotations

import time
from typing import Any

from finetune_studio.db.connection import cursor, new_id, row_to_dict

_ALLOWED_MANIFEST_KEYS = {
    "datasets",        # list[dict] {id, name, data_path, qa_count}
    "source_ids",      # list[str] qa source ids selected for the dataset
    "rag_corpora",     # list[dict] {build_id, rag_id, corpus_dir, doc_count, chunk_count}
    "training_runs",   # list[dict] {run_id, status, final_loss, output_path}
    "base_model",      # str
    "suites",          # list[dict] {suite_path, case_count}
}


def _clean_manifest(manifest: dict[str, Any] | None) -> str:
    if not manifest:
        return "{}"
    cleaned = {k: v for k, v in manifest.items() if k in _ALLOWED_MANIFEST_KEYS}
    import json
    return json.dumps(cleaned, ensure_ascii=False)


def create_version(project_id: str, *, version_number: int | None = None,
                   label: str = "", notes: str = "",
                   parent_version_id: str | None = None,
                   manifest: dict[str, Any] | None = None) -> dict:
    """Register a new immutable project version.

    `version_number` is derived (max+1) when omitted, so numbers stay
    monotonic per project even under concurrent saves.
    """
    vid = new_id()
    now = time.time()
    manifest_json = _clean_manifest(manifest)
    with cursor() as c:
        if version_number is None:
            row = c.execute(
                "SELECT COALESCE(MAX(version_number), 0) AS mx FROM project_versions "
                "WHERE project_id = ?", (project_id,)).fetchone()
            version_number = int(row["mx"]) + 1 if row else 1
        c.execute(
            "INSERT INTO project_versions "
            "(id, project_id, version_number, label, notes, manifest_json, "
            " parent_version_id, created_at) VALUES (?,?,?,?,?,?,?,?)",
            (vid, project_id, version_number, label, notes, manifest_json,
             parent_version_id, now))
    v = get_version(vid)
    assert v is not None
    return v


def get_version(vid: str) -> dict | None:
    with cursor() as c:
        r = c.execute("SELECT * FROM project_versions WHERE id = ?", (vid,)).fetchone()
    return row_to_dict(r)


def get_version_by_number(project_id: str, version_number: int) -> dict | None:
    with cursor() as c:
        r = c.execute(
            "SELECT * FROM project_versions WHERE project_id = ? AND version_number = ?",
            (project_id, version_number)).fetchone()
    return row_to_dict(r)


def list_versions(project_id: str) -> list[dict]:
    with cursor() as c:
        rows = c.execute(
            "SELECT * FROM project_versions WHERE project_id = ? "
            "ORDER BY version_number DESC", (project_id,)).fetchall()
    out = []
    for r in rows:
        d = row_to_dict(r)
        if d:
            out.append(d)
    return out


def latest_version(project_id: str) -> dict | None:
    vs = list_versions(project_id)
    return vs[0] if vs else None


def delete_version(vid: str) -> bool:
    """Delete a manifest row only — never the artifacts it references."""
    with cursor() as c:
        c.execute("DELETE FROM project_versions WHERE id = ?", (vid,))
        return c.rowcount > 0


def version_lineage(project_id: str, vid: str) -> list[dict]:
    """Walk parent_version_id links from `vid` back to the root, oldest first."""
    chain: list[dict] = []
    seen: set[str] = set()
    cur: dict | None = get_version(vid)
    while cur and cur.get("id") not in seen:
        seen.add(str(cur.get("id")))
        chain.append(cur)
        parent_id = cur.get("parent_version_id")
        cur = get_version(str(parent_id)) if parent_id else None
        if cur and str(cur.get("project_id")) != project_id:
            break
    chain.reverse()
    return chain
