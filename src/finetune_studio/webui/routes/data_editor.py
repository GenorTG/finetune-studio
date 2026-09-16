"""Data Editor API — project-scoped JSONL row editing.

Endpoints mounted under /api/data-editor in app.py.
"""

from __future__ import annotations

import json
from pathlib import Path

from fastapi import APIRouter, HTTPException, Request
from fastapi.templating import Jinja2Templates

from finetune_studio import db
from finetune_studio import __release_channel__ as RELEASE_CHANNEL
from finetune_studio import __version__ as APP_VERSION
from finetune_studio.config import settings
from finetune_studio.training.data import load_jsonl, save_jsonl

router = APIRouter()
templates = Jinja2Templates(directory=str(Path(__file__).parent.parent / "templates"))
templates.env.globals["app_version"] = APP_VERSION
templates.env.globals["release_channel"] = RELEASE_CHANNEL


def _has_traversal(raw: str) -> bool:
    """True if any path component is ``..`` (rejects encoded traversal too after decode)."""
    return ".." in Path(raw).parts


def _project_root(pid: str) -> Path:
    """Filesystem root for a project's stored artifacts (datasets live under this)."""
    return (Path(settings.db_path).parent / "projects" / pid).resolve()


def _normalize_raw_path(dataset: str) -> Path:
    """Map stored/API path forms to a Path without double-prefixing data_dir.

    Handles:
    - absolute paths as-is
    - relative under data_dir (``datasets/foo.jsonl`` → ``{data_dir}/datasets/foo.jsonl``)
    - already-prefixed forms (``data/projects/...`` when data_dir is ``data``) without
      joining again into ``data/data/projects/...``
    """
    raw = (dataset or "").strip()
    if not raw:
        raise HTTPException(400, "dataset required")
    if _has_traversal(raw):
        raise HTTPException(400, "path traversal not allowed")

    p = Path(raw)
    if p.is_absolute():
        return p

    data_root = Path(settings.data_dir)
    root_posix = data_root.as_posix().rstrip("/")
    # Stored relative form already includes data_dir (export writes str(Path) under db parent).
    if raw == root_posix or raw.startswith(root_posix + "/"):
        return Path(raw)

    # Cwd-relative path that already exists (common when data_dir == "data" and cwd is repo).
    if p.exists():
        return p

    return data_root / raw


def _paths_equal(a: Path, b: Path) -> bool:
    """Compare paths, preferring resolve() but falling back to string equality."""
    if str(a) == str(b):
        return True
    try:
        return a.resolve() == b.resolve()
    except OSError:
        return False


def _is_under(path: Path, root: Path) -> bool:
    try:
        path.resolve().relative_to(root.resolve())
        return True
    except (ValueError, OSError):
        return False


def _assert_project_scope(pid: str, path: Path) -> None:
    """Reject paths outside this project unless registered to the project."""
    from finetune_studio.db.datasets import list_datasets

    for ds in list_datasets(pid):
        if _paths_equal(Path(ds["data_path"]), path):
            return

    if _is_under(path, _project_root(pid)):
        return

    raise HTTPException(403, "dataset path outside project scope")


def _resolve_path(dataset: str, *, pid: str) -> Path:
    """Resolve a dataset path for a project with safe normalization and scoping."""
    from finetune_studio.db.datasets import get_dataset_by_path

    raw = (dataset or "").strip()
    if not raw:
        raise HTTPException(400, "dataset required")
    if _has_traversal(raw):
        raise HTTPException(400, "path traversal not allowed")

    registered = get_dataset_by_path(pid, raw)
    if registered:
        path = _normalize_raw_path(registered["data_path"])
        _assert_project_scope(pid, path)
        return path

    path = _normalize_raw_path(raw)
    _assert_project_scope(pid, path)
    return path


def _load(dataset: str, *, pid: str) -> list[dict]:
    path = _resolve_path(dataset, pid=pid)
    if not path.exists():
        raise HTTPException(404, f"File not found: {dataset}")
    return load_jsonl(str(path))


def _save(rows: list[dict], dataset: str, *, pid: str) -> None:
    path = _resolve_path(dataset, pid=pid)
    path.parent.mkdir(parents=True, exist_ok=True)
    save_jsonl(rows, str(path))


# ── Preview (paginated row list) ────────────────────────────────────────

@router.get("/projects/{pid}/preview")
async def preview(pid: str, dataset: str = "", limit: int = 50, offset: int = 0):
    project = db.get_project(pid)
    if not project:
        raise HTTPException(404, "Project not found")
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
    body = await request.json()
    dataset = body.get("dataset", "")
    index = body.get("index")
    new_row = body.get("row")
    if index is None or new_row is None or not dataset:
        raise HTTPException(400, "dataset, index, and row required")
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
    rows = _load(dataset, pid=pid)
    if index < 0 or index >= len(rows):
        raise HTTPException(404, "Row index out of range")
    rows.pop(index)
    _save(rows, dataset, pid=pid)
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
    _save(rows, dataset, pid=pid)
    return {"ok": True, "total": len(rows)}
