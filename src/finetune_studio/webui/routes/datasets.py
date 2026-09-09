"""Routes for project-scoped training datasets (jsonl files).

Endpoints:
  GET    /api/projects/{pid}/datasets            — list registered datasets
  GET    /api/projects/{pid}/datasets/{did}      — single dataset metadata
  POST   /api/projects/{pid}/datasets            — register an existing file (json body)
  POST   /api/projects/{pid}/datasets/upload     — multipart upload; saves + registers
  PATCH  /api/projects/{pid}/datasets/{did}      — rename / update stats
  DELETE /api/projects/{pid}/datasets/{did}      — unregister (+ optionally delete file)
"""

from __future__ import annotations

import logging
import shutil
from pathlib import Path

from fastapi import APIRouter, File, Request, UploadFile
from fastapi.responses import JSONResponse

from finetune_studio import db
from finetune_studio.db.datasets import count_qa_pairs, datasets_dir

log = logging.getLogger(__name__)

router = APIRouter()


@router.get("/projects/{pid}/datasets")
async def list_datasets_route(pid: str):
    """List all registered datasets for a project."""
    proj = db.get_project(pid)
    if not proj:
        return JSONResponse({"error": "project not found"}, status_code=404)
    return {"datasets": db.list_datasets(pid)}


@router.get("/projects/{pid}/datasets/{did}")
async def get_dataset_route(pid: str, did: str):
    ds = db.get_dataset(did)
    if not ds or ds.get("project_id") != pid:
        return JSONResponse({"error": "not found"}, status_code=404)
    # Refresh qa_count + size_bytes if file changed (best-effort).
    try:
        p = Path(ds["data_path"])
        if p.exists():
            size_now = p.stat().st_size
            if size_now != ds.get("size_bytes", 0):
                qa_now = count_qa_pairs(str(p))
                db.update_dataset(did, size_bytes=size_now, qa_count=qa_now)
                ds = db.get_dataset(did) or ds
    except Exception:  # noqa: BLE001
        log.warning("dataset stat refresh failed", exc_info=True)
    return ds


@router.post("/projects/{pid}/datasets")
async def register_existing_route(pid: str, request: Request):
    """Register an existing file on disk (e.g. written by data-prep export)."""
    body = await request.json()
    data_path = body.get("data_path", "")
    if not data_path:
        return JSONResponse({"error": "data_path required"}, status_code=400)
    p = Path(data_path)
    if not p.exists():
        return JSONResponse({"error": f"file not found: {data_path}"}, status_code=404)
    # Dedup by path
    existing = db.get_dataset_by_path(pid, str(p))
    if existing:
        return existing
    name = body.get("name") or p.stem
    source = body.get("source", "data-prep-export")
    qa_count = count_qa_pairs(str(p))
    return db.create_dataset(
        project_id=pid,
        name=name,
        data_path=str(p),
        source=source,
        qa_count=qa_count,
        size_bytes=p.stat().st_size,
    )


@router.post("/projects/{pid}/datasets/upload")
async def upload_dataset_route(pid: str, file: UploadFile = File(...)):
    """Multipart upload: save to the project's datasets dir and register."""
    proj = db.get_project(pid)
    if not proj:
        return JSONResponse({"error": "project not found"}, status_code=404)
    raw = await file.read()
    if not raw:
        return JSONResponse({"error": "empty upload"}, status_code=400)
    # Coerce .json / .txt / etc → .jsonl so downstream loaders recognise it.
    fname = file.filename or "uploaded.jsonl"
    p = Path(fname)
    stem, suf = p.stem, p.suffix.lower()
    if suf != ".jsonl":
        suf = ".jsonl"
    out_dir = datasets_dir(pid)
    target = out_dir / f"{stem}{suf}"
    # Avoid clobbering: append suffix if file exists.
    counter = 2
    while target.exists():
        target = out_dir / f"{stem}-{counter}{suf}"
        counter += 1
    target.write_bytes(raw)
    qa_count = count_qa_pairs(str(target))
    return db.create_dataset(
        project_id=pid,
        name=target.stem,
        data_path=str(target),
        source="upload",
        qa_count=qa_count,
        size_bytes=len(raw),
    )


@router.patch("/projects/{pid}/datasets/{did}")
async def patch_dataset_route(pid: str, did: str, request: Request):
    body = await request.json()
    ds = db.get_dataset(did)
    if not ds or ds.get("project_id") != pid:
        return JSONResponse({"error": "not found"}, status_code=404)
    return db.update_dataset(did, **{k: v for k, v in body.items() if k in {"name"}})


@router.delete("/projects/{pid}/datasets/{did}")
async def delete_dataset_route(pid: str, did: str, remove_file: bool = False):
    ds = db.get_dataset(did)
    if not ds or ds.get("project_id") != pid:
        return JSONResponse({"error": "not found"}, status_code=404)
    return db.delete_dataset(did, remove_file=remove_file)
