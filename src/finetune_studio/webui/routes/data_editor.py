"""Data Editor API — project-scoped JSONL row editing.

Endpoints mounted under /api/data-editor in app.py.
"""

from __future__ import annotations

import json
from pathlib import Path

from fastapi import APIRouter, HTTPException, Request
from fastapi.templating import Jinja2Templates

from finetune_studio import db
from finetune_studio.config import settings
from finetune_studio.training.data import load_jsonl, save_jsonl

router = APIRouter()
templates = Jinja2Templates(directory=str(Path(__file__).parent.parent / "templates"))


def _resolve_path(dataset: str) -> Path:
    """Resolve a dataset path: absolute as-is, relative under data_dir."""
    p = Path(dataset)
    if p.is_absolute():
        return p
    return Path(settings.data_dir) / dataset


def _load(dataset: str) -> list[dict]:
    path = _resolve_path(dataset)
    if not path.exists():
        raise HTTPException(404, f"File not found: {dataset}")
    return load_jsonl(str(path))


def _save(rows: list[dict], dataset: str) -> None:
    path = _resolve_path(dataset)
    path.parent.mkdir(parents=True, exist_ok=True)
    save_jsonl(rows, str(path))


# ── Preview (paginated row list) ────────────────────────────────────────

@router.get("/projects/{pid}/preview")
async def preview(pid: str, dataset: str = "", limit: int = 50, offset: int = 0):
    project = db.get_project(pid)
    if not project:
        raise HTTPException(404, "Project not found")
    rows = _load(dataset)
    total = len(rows)
    page = rows[offset : offset + limit]
    return {"rows": page, "total": total, "offset": offset, "limit": limit}


# ── Single row ──────────────────────────────────────────────────────────

@router.get("/projects/{pid}/row")
async def get_row(pid: str, dataset: str = "", index: int = 0):
    project = db.get_project(pid)
    if not project:
        raise HTTPException(404, "Project not found")
    rows = _load(dataset)
    if index < 0 or index >= len(rows):
        raise HTTPException(404, "Row index out of range")
    return rows[index]


@router.patch("/projects/{pid}/row")
async def update_row(pid: str, request: Request):
    project = db.get_project(pid)
    if not project:
        raise HTTPException(404, "Project not found")
    body = await request.json()
    dataset = body.get("dataset", "")
    index = body.get("index")
    new_row = body.get("row")
    if index is None or new_row is None or not dataset:
        raise HTTPException(400, "dataset, index, and row required")
    rows = _load(dataset)
    if index < 0 or index >= len(rows):
        raise HTTPException(404, "Row index out of range")
    rows[index] = new_row
    _save(rows, dataset)
    db.record_review(pid, dataset, index, "edited", json.dumps(new_row, ensure_ascii=False))
    return {"ok": True}


# ── Approve / Reject ────────────────────────────────────────────────────

@router.post("/projects/{pid}/approve")
async def approve_row(pid: str, request: Request):
    project = db.get_project(pid)
    if not project:
        raise HTTPException(404, "Project not found")
    body = await request.json()
    dataset = body.get("dataset", "")
    index = body.get("index")
    if index is None or not dataset:
        raise HTTPException(400, "dataset and index required")
    db.record_review(pid, dataset, int(index), "approved")
    return {"ok": True}


@router.post("/projects/{pid}/reject")
async def reject_row(pid: str, request: Request):
    project = db.get_project(pid)
    if not project:
        raise HTTPException(404, "Project not found")
    body = await request.json()
    dataset = body.get("dataset", "")
    index = body.get("index")
    if index is None or not dataset:
        raise HTTPException(400, "dataset and index required")
    db.record_review(pid, dataset, int(index), "rejected")
    return {"ok": True}


# ── Delete row ──────────────────────────────────────────────────────────

@router.delete("/projects/{pid}/row")
async def delete_row(pid: str, request: Request):
    project = db.get_project(pid)
    if not project:
        raise HTTPException(404, "Project not found")
    body = await request.json()
    dataset = body.get("dataset", "")
    index = body.get("index")
    if index is None or not dataset:
        raise HTTPException(400, "dataset and index required")
    rows = _load(dataset)
    if index < 0 or index >= len(rows):
        raise HTTPException(404, "Row index out of range")
    rows.pop(index)
    _save(rows, dataset)
    db.record_review(pid, dataset, int(index), "deleted")
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
    body = await request.json()
    dataset = body.get("dataset", "")
    rows = body.get("rows")
    if rows is None or not dataset:
        raise HTTPException(400, "dataset and rows required")
    _save(rows, dataset)
    return {"ok": True, "total": len(rows)}
