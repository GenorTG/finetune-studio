"""Routes for project-scoped training datasets (jsonl files).

Endpoints:
  GET    /api/projects/{pid}/datasets            — list registered datasets
  GET    /api/projects/{pid}/datasets/{did}      — single dataset metadata
  POST   /api/projects/{pid}/datasets            — register an existing file (json body)
  POST   /api/projects/{pid}/datasets/upload     — multipart upload; saves + registers
  PATCH  /api/projects/{pid}/datasets/{did}      — rename (only ``name`` is patchable)
  DELETE /api/projects/{pid}/datasets/{did}      — unregister (+ optionally delete file)
"""

from __future__ import annotations

import logging
from pathlib import Path

from fastapi import APIRouter, File, HTTPException, Request, UploadFile
from fastapi.responses import JSONResponse

from finetune_studio import db
from finetune_studio.data.converter import records_from_upload, write_jsonl
from finetune_studio.data.dataset_health import check_dataset, dedupe_dataset
from finetune_studio.data.fs import file_library as fl
from finetune_studio.data.fs.paths import resolve_in_project
from finetune_studio.db.datasets import (
    count_qa_pairs,
    datasets_dir,
    get_dataset_by_path,
)

log = logging.getLogger(__name__)

router = APIRouter()


@router.get("/projects/{pid}/datasets")
async def list_datasets_route(pid: str):
    """List all registered datasets for a project."""
    proj = (
        db.get_project(pid)
    )
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
    except Exception:
        log.warning("dataset stat refresh failed", exc_info=True)
    return ds


@router.post("/projects/{pid}/datasets")
async def register_existing_route(pid: str, request: Request):
    """Register an existing file on disk (e.g. written by data-prep export).

    Body accepts either ``data_path`` (path inside the project dir) **or** ``file_id``
    (a row id from ``project_files``; we resolve to ``stored_path``).
    """
    if not db.get_project(pid):
        return JSONResponse({"error": "project not found"}, status_code=404)
    body = await request.json()
    if not isinstance(body, dict):
        return JSONResponse({"error": "JSON object body required"}, status_code=400)
    data_path = body.get("data_path") or ""
    # Fallback: WebUI data-prep sends {file_id}; resolve via project_files.
    if not data_path:
        fid = body.get("file_id")
        if fid:
            # Look up the file's actual project_id directly (bypasses
            # fl.get_file's pid filter so we can return 403 for cross-project).
            owner_pid: str | None = None
            with db.cursor() as _c:
                _row = _c.execute(
                    "SELECT project_id FROM project_files WHERE id = ?", (fid,)
                ).fetchone()
            if _row:
                owner_pid = _row["project_id"]
            if owner_pid and owner_pid != pid:
                return JSONResponse(
                    {"error": f"file {fid} belongs to another project"},
                    status_code=403,
                )
            if owner_pid == pid:
                # The on-disk path lives in file_versions (not project_files).
                versions = fl.list_versions(pid, fid)
                if versions:
                    data_path = versions[0].get("raw_path") or ""
    if not data_path:
        return JSONResponse(
            {"error": "data_path or file_id required"}, status_code=400
        )
    try:
        p = resolve_in_project(pid, data_path, what="data_path")
    except HTTPException as exc:
        return JSONResponse({"error": exc.detail}, status_code=exc.status_code)
    if not p.is_file():
        return JSONResponse({"error": f"file not found: {data_path}"}, status_code=404)
    # Dedup by path
    existing = get_dataset_by_path(pid, str(p))
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
async def upload_dataset_route(pid: str, file: UploadFile = File(...)):  # noqa: B008
    """Multipart upload: convert .jsonl/.json/.csv to training JSONL and register it."""
    proj = db.get_project(pid)
    if not proj:
        return JSONResponse({"error": "project not found"}, status_code=404)
    raw = await file.read()
    if not raw:
        return JSONResponse({"error": "empty upload"}, status_code=400)
    fname = file.filename or "uploaded.jsonl"
    try:
        records = records_from_upload(raw, fname)
    except ValueError as e:
        return JSONResponse({"error": f"{Path(fname).name}: {e}"}, status_code=400)
    target = _unique_dataset_path(pid, Path(fname).stem)
    write_jsonl(records, target)
    return db.create_dataset(
        project_id=pid,
        name=target.stem,
        data_path=str(target),
        source="upload",
        qa_count=len(records),
        size_bytes=target.stat().st_size,
    )


def _unique_dataset_path(pid: str, stem: str) -> Path:
    """``<datasets>/<stem>.jsonl``, suffixed ``-2``, ``-3``… so nothing is clobbered."""
    out_dir = datasets_dir(pid)
    target = out_dir / f"{stem}.jsonl"
    counter = 2
    while target.exists():
        target = out_dir / f"{stem}-{counter}.jsonl"
        counter += 1
    return target


def _project_dataset(pid: str, did: str) -> dict | None:
    ds = db.get_dataset(did)
    return ds if ds and ds.get("project_id") == pid else None


@router.get("/projects/{pid}/datasets/{did}/health")
async def dataset_health_route(pid: str, did: str):
    """Plain-language health report for one dataset (see ``data.dataset_health``)."""
    ds = _project_dataset(pid, did)
    if not ds:
        return JSONResponse({"error": "dataset not found in this project"}, status_code=404)
    path = Path(ds["data_path"])
    if not path.is_file():
        return JSONResponse({"error": f"dataset file is missing: {path.name}"}, status_code=404)
    return check_dataset(path)


@router.post("/projects/{pid}/datasets/{did}/dedup")
async def dataset_dedup_route(pid: str, did: str):
    """Write a duplicate-free copy as a new dataset; the original is left untouched."""
    ds = _project_dataset(pid, did)
    if not ds:
        return JSONResponse({"error": "dataset not found in this project"}, status_code=404)
    src = Path(ds["data_path"])
    if not src.is_file():
        return JSONResponse({"error": f"dataset file is missing: {src.name}"}, status_code=404)
    target = _unique_dataset_path(pid, f"{src.stem}-dedup")
    kept, removed = dedupe_dataset(src, target)
    if not removed:
        target.unlink(missing_ok=True)
        return JSONResponse({"error": "no duplicates to remove"}, status_code=400)
    new = db.create_dataset(
        project_id=pid,
        name=target.stem,
        data_path=str(target),
        source="dedup",
        qa_count=kept,
        size_bytes=target.stat().st_size,
    )
    return {**new, "removed": removed}


@router.patch("/projects/{pid}/datasets/{did}")
async def patch_dataset_route(pid: str, did: str, request: Request):
    body = await request.json()
    ds = db.get_dataset(did)
    if not ds or ds.get("project_id") != pid:
        return JSONResponse({"error": "not found"}, status_code=404)
    name = body.get("name") if isinstance(body, dict) else None
    if not isinstance(name, str) or not name.strip():
        return JSONResponse({"error": "name (non-empty string) required"}, status_code=400)
    return db.update_dataset(did, name=name.strip())


@router.delete("/projects/{pid}/datasets/{did}")
async def delete_dataset_route(pid: str, did: str, remove_file: bool = False):
    ds = db.get_dataset(did)
    if not ds or ds.get("project_id") != pid:
        return JSONResponse({"error": "not found"}, status_code=404)
    return db.delete_dataset(did, remove_file=remove_file)
