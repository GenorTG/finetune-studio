"""Data Editor API — project-scoped JSONL row editing.

Endpoints mounted under /api/data-editor in app.py.
"""

from __future__ import annotations

import json
from pathlib import Path

from fastapi import APIRouter, HTTPException, Request

from finetune_studio import db
from finetune_studio.training.data import load_jsonl, save_jsonl

router = APIRouter()


def _resolve_path(dataset: str, *, pid: str) -> Path:
    """Resolve a dataset reference to a path inside project ``pid`` (400/403 otherwise)."""
    from finetune_studio.data.fs.paths import resolve_in_project
    from finetune_studio.db.datasets import get_dataset_by_path

    raw = (dataset or "").strip()
    if not raw:
        raise HTTPException(400, "dataset required")
    registered = get_dataset_by_path(pid, raw)
    return resolve_in_project(pid, registered["data_path"] if registered else raw, what="dataset")


def _load(dataset: str, *, pid: str) -> list[dict]:
    path = _resolve_path(dataset, pid=pid)
    if not path.exists():
        raise HTTPException(404, f"File not found: {dataset}")
    return load_jsonl(str(path))


def _save(rows: list[dict], dataset: str, *, pid: str) -> None:
    path = _resolve_path(dataset, pid=pid)
    path.parent.mkdir(parents=True, exist_ok=True)
    save_jsonl(rows, str(path))


def _parse_index(raw: object) -> int:
    """Coerce a JSON ``index`` to int; 400 on bool/str junk instead of a 500."""
    if isinstance(raw, bool):
        raise HTTPException(400, "index must be an integer")
    try:
        return int(raw)  # type: ignore[call-overload]
    except (TypeError, ValueError):
        raise HTTPException(400, "index must be an integer") from None


async def _json_body(request: Request) -> dict:
    body = await request.json()
    if not isinstance(body, dict):
        raise HTTPException(400, "JSON object body required")
    return body


# ── Preview (paginated row list) ────────────────────────────────────────

@router.get("/projects/{pid}/preview")
async def preview(pid: str, dataset: str = "", limit: int = 50, offset: int = 0):
    project = db.get_project(pid)
    if not project:
        raise HTTPException(404, "Project not found")
    offset = max(0, offset)
    limit = max(0, limit)
    rows = _load(dataset, pid=pid)
    total = len(rows)
    page = rows[offset : offset + limit]
    return {"rows": page, "total": total, "offset": offset, "limit": limit}


# ── Single row ──────────────────────────────────────────────────────────

@router.get("/projects/{pid}/row")
async def get_row(pid: str, dataset: str = "", index: int = 0):
    project = db.get_project(pid)
    if not project:
        raise HTTPException(404, "Project not found")
    rows = _load(dataset, pid=pid)
    if index < 0 or index >= len(rows):
        raise HTTPException(404, "Row index out of range")
    return rows[index]


@router.patch("/projects/{pid}/row")
async def update_row(pid: str, request: Request):
    project = db.get_project(pid)
    if not project:
        raise HTTPException(404, "Project not found")
    body = await _json_body(request)
    dataset = body.get("dataset", "")
    index = body.get("index")
    new_row = body.get("row")
    if index is None or new_row is None or not dataset:
        raise HTTPException(400, "dataset, index, and row required")
    if not isinstance(new_row, dict):
        raise HTTPException(400, "row must be an object")
    index = _parse_index(index)
    rows = _load(dataset, pid=pid)
    if index < 0 or index >= len(rows):
        raise HTTPException(404, "Row index out of range")
    rows[index] = new_row
    _save(rows, dataset, pid=pid)
    db.record_review(pid, dataset, index, "edited", json.dumps(new_row, ensure_ascii=False))
    return {"ok": True}


# ── Approve / Reject ────────────────────────────────────────────────────

@router.post("/projects/{pid}/approve")
async def approve_row(pid: str, request: Request):
    project = db.get_project(pid)
    if not project:
        raise HTTPException(404, "Project not found")
    body = await _json_body(request)
    dataset = body.get("dataset", "")
    index = body.get("index")
    if index is None or not dataset:
        raise HTTPException(400, "dataset and index required")
    db.record_review(pid, dataset, _parse_index(index), "approved")
    return {"ok": True}


@router.post("/projects/{pid}/reject")
async def reject_row(pid: str, request: Request):
    project = db.get_project(pid)
    if not project:
        raise HTTPException(404, "Project not found")
    body = await _json_body(request)
    dataset = body.get("dataset", "")
    index = body.get("index")
    if index is None or not dataset:
        raise HTTPException(400, "dataset and index required")
    db.record_review(pid, dataset, _parse_index(index), "rejected")
    return {"ok": True}


# ── Delete row ──────────────────────────────────────────────────────────

@router.delete("/projects/{pid}/row")
async def delete_row(pid: str, request: Request):
    project = db.get_project(pid)
    if not project:
        raise HTTPException(404, "Project not found")
    body = await _json_body(request)
    dataset = body.get("dataset", "")
    index = body.get("index")
    if index is None or not dataset:
        raise HTTPException(400, "dataset and index required")
    index = _parse_index(index)
    rows = _load(dataset, pid=pid)
    if index < 0 or index >= len(rows):
        raise HTTPException(404, "Row index out of range")
    rows.pop(index)
    _save(rows, dataset, pid=pid)
    db.record_review(pid, dataset, index, "deleted")
    return {"ok": True, "total": len(rows)}


# ── Review history ──────────────────────────────────────────────────────

@router.get("/projects/{pid}/review")
async def review_list(pid: str, dataset: str = ""):
    project = db.get_project(pid)
    if not project:
        raise HTTPException(404, "Project not found")
    return db.list_review(pid, dataset)


# ── Batch save ──────────────────────────────────────────────────────────

@router.post("/projects/{pid}/save")
async def batch_save(pid: str, request: Request):
    project = db.get_project(pid)
    if not project:
        raise HTTPException(404, "Project not found")
    body = await _json_body(request)
    dataset = body.get("dataset", "")
    rows = body.get("rows")
    if rows is None or not dataset:
        raise HTTPException(400, "dataset and rows required")
    if not isinstance(rows, list) or not all(isinstance(r, dict) for r in rows):
        raise HTTPException(400, "rows must be a list of objects")
    _save(rows, dataset, pid=pid)
    return {"ok": True, "total": len(rows)}
