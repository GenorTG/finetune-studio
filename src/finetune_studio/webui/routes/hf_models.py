"""Routes for browsing + downloading HuggingFace models.

UI: LM Studio-style model explorer. Search the HF Hub, filter by task,
inspect model cards, download a specific file or full repo to a local cache.

Local cache: ~/.finetune-studio/hf_models/<repo_id>/<file>  (NOT the default
HF hub cache, because we want a single user-facing directory listing).
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import shutil
import subprocess
import threading
import time
import uuid
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Optional

from fastapi import APIRouter, BackgroundTasks, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel

log = logging.getLogger(__name__)
router = APIRouter()

_LOCAL = Path.home() / ".finetune-studio" / "hf_models"
_DOWNLOADS: dict[str, dict] = {}  # job_id -> progress dict (process-wide)


# ── DTOs ────────────────────────────────────────────────────────────────

class SearchRequest(BaseModel):
    query: str = ""
    task: Optional[str] = "text-generation"  # text-generation, image-text-to-text, ...
    library: Optional[str] = None           # transformers, sentence-transformers, ...
    sort: str = "downloads"                  # downloads | likes | trending
    limit: int = 24
    full: bool = False                       # if True, no client-side filtering


# ── Local model library ────────────────────────────────────────────────

def _local_models() -> list[dict]:
    """Walk the local cache and return one entry per (repo_id, branch_or_file)."""
    _LOCAL.mkdir(parents=True, exist_ok=True)
    out = []
    for repo_dir in sorted(_LOCAL.iterdir()):
        if not repo_dir.is_dir():
            continue
        # Each repo can have a "snapshots" subdir (HF cache layout) or just files.
        size = sum(p.stat().st_size for p in repo_dir.rglob("*") if p.is_file())
        files = []
        for p in sorted(repo_dir.rglob("*")):
            if p.is_file():
                files.append({
                    "path": str(p.relative_to(_LOCAL)),
                    "size_bytes": p.stat().st_size,
                })
        out.append({
            "repo_id": repo_dir.name,
            "path": str(repo_dir),
            "size_bytes": size,
            "file_count": len(files),
            "files": files[:50],  # cap listing
        })
    return out


# ── Hub search ─────────────────────────────────────────────────────────

def _search_hf(req: SearchRequest) -> list[dict]:
    """Query the HuggingFace Hub."""
    from huggingface_hub import HfApi
    api = HfApi()
    sort = {"downloads": "downloads", "likes": "likes",
            "trending": "trendingScore"}.get(req.sort, "downloads")
    try:
        # Use the real Hub text search when a query is present — this does a
        # proper fuzzy/full-text match server-side instead of fetching a tiny
        # window of popular models and substring-filtering locally.
        models = api.list_models(
            search=req.query or None,
            pipeline_tag=req.task or None,
            sort=sort,
            limit=req.limit * 2 + 10,
        )
        out = []
        for m in models:
            mid = m.modelId
            if req.query and req.query.lower() not in mid.lower():
                continue
            out.append({
                "repo_id": mid,
                "downloads": getattr(m, "downloads", 0) or 0,
                "likes": getattr(m, "likes", 0) or 0,
                "tags": getattr(m, "tags", []) or [],
                "last_modified": str(getattr(m, "lastModified", "")),
                "private": getattr(m, "private", False),
            })
            if len(out) >= req.limit:
                break
        return out
    except Exception as e:
        log.warning("HF search failed: %s", e)
        return []


def _model_info(repo_id: str) -> Optional[dict]:
    from huggingface_hub import HfApi
    api = HfApi()
    try:
        info = api.model_info(repo_id)
        files = []
        try:
            siblings = api.list_repo_files(repo_id, repo_type="model")
        except Exception:
            siblings = []
        for s in siblings:
            files.append({"path": s, "size": None})
        return {
            "repo_id": info.modelId,
            "downloads": getattr(info, "downloads", 0) or 0,
            "likes": getattr(info, "likes", 0) or 0,
            "tags": getattr(info, "tags", []) or [],
            "pipeline_tag": getattr(info, "pipeline_tag", None),
            "library_name": getattr(info, "library_name", None),
            "private": getattr(info, "private", False),
            "last_modified": str(getattr(info, "lastModified", "")),
            "card_data": {
                "description": (getattr(info, "card_data", {}) or {}).get("description", ""),
                "license": (getattr(info, "card_data", {}) or {}).get("license", ""),
                "tags": (getattr(info, "card_data", {}) or {}).get("tags", []),
            } if getattr(info, "card_data", None) else {},
            "files": files,
        }
    except Exception as e:
        log.warning("HF model_info failed: %s", e)
        return None


# ── Routes ──────────────────────────────────────────────────────────────

@router.get("/hf/search")
async def hf_search(q: str = "", task: str = "text-generation",
                   library: Optional[str] = None,
                   sort: str = "downloads", limit: int = 24):
    req = SearchRequest(query=q, task=task, library=library, sort=sort, limit=limit)
    return {"results": _search_hf(req), "task": task, "query": q}


@router.get("/hf/info/{repo_id:path}")
async def hf_info(repo_id: str):
    info = _model_info(repo_id)
    if info is None:
        return JSONResponse({"error": "model not found or fetch failed"}, status_code=404)
    return info


class DownloadRequest(BaseModel):
    repo_id: str
    filename: Optional[str] = None  # if set, download just that file; else full repo
    revision: str = "main"


@router.post("/hf/download")
async def hf_download(req: DownloadRequest, background: BackgroundTasks):
    """Start a background download; return immediately with a job_id.
    Track progress via /hf/download/progress?job_id=..."""
    from finetune_studio import db
    job_row = db.create_hf_download(repo_id=req.repo_id, filename=req.filename or "")
    job_id = job_row["id"]
    _DOWNLOADS[job_id] = {
        "job_id": job_id,
        "status": "queued",
        "repo_id": req.repo_id,
        "filename": req.filename,
        "started_at": job_row.get("started_at"),
        "created_at": job_row.get("created_at"),
        "bytes_done": 0,
        "bytes_total": None,
        "path": None,
        "error": None,
    }
    background.add_task(_download_worker, job_id, req.repo_id, req.filename, req.revision)
    return {"ok": True, "job_id": job_id, "started_at": _DOWNLOADS[job_id]["started_at"]}


@router.get("/hf/download/progress")
async def hf_download_progress(job_id: str):
    # Prefer in-memory dict (freshest). Fall back to DB row for restart
    # survivors + historical jobs.
    p = _DOWNLOADS.get(job_id)
    if p:
        return p
    from finetune_studio import db
    row = db.get_hf_download(job_id)
    if row is None:
        return JSONResponse({"error": "unknown job_id"}, status_code=404)
    return row


@router.post("/hf/download/cancel")
async def hf_download_cancel(job_id: str):
    """Mark a job as cancelled (best-effort — child subprocesses may continue briefly)."""
    p = _DOWNLOADS.get(job_id)
    if not p:
        # Maybe a historical DB row exists.
        from finetune_studio import db
        row = db.get_hf_download(job_id)
        if row is None:
            return JSONResponse({"error": "unknown job_id"}, status_code=404)
        db.mark_hf_download_cancelled(job_id)
        return {"ok": True, "job_id": job_id, "status": "cancelled"}
    p["status"] = "cancelled"
    p["error"] = "user cancelled"
    try:
        from finetune_studio import db
        db.mark_hf_download_cancelled(job_id)
    except Exception:  # noqa: BLE001
        pass
    return {"ok": True, "job_id": job_id, "status": "cancelled"}


def _download_worker(job_id: str, repo_id: str, filename: Optional[str], revision: str):
    """Background download via huggingface_hub.snapshot_download or hf_hub_download."""
    from finetune_studio import db
    try:
        from huggingface_hub import snapshot_download, hf_hub_download
        db.mark_hf_download_running(job_id)
        if job_id in _DOWNLOADS:
            _DOWNLOADS[job_id].update({"status": "downloading", "bytes_done": 0})
        dest_root = _LOCAL / repo_id.replace("/", "__")
        dest_root.mkdir(parents=True, exist_ok=True)
        if filename:
            # Single-file download
            local_path = hf_hub_download(
                repo_id=repo_id, filename=filename, revision=revision,
                local_dir=str(dest_root),
            )
            size = os.path.getsize(local_path) if os.path.exists(local_path) else 0
            db.mark_hf_download_done(job_id, path=local_path,
                                     bytes_total=size, bytes_done=size)
            if job_id in _DOWNLOADS:
                _DOWNLOADS[job_id].update({
                    "status": "completed",
                    "bytes_total": size, "bytes_done": size,
                    "path": local_path,
                })
        else:
            # Snapshot the whole repo
            local_path = snapshot_download(
                repo_id=repo_id, revision=revision,
                local_dir=str(dest_root),
                # tqdm progress gets noisy in logs; we just mark progress at the end
            )
            total = 0
            for p in Path(local_path).rglob("*"):
                if p.is_file():
                    total += p.stat().st_size
            db.mark_hf_download_done(job_id, path=local_path,
                                     bytes_total=total, bytes_done=total)
            if job_id in _DOWNLOADS:
                _DOWNLOADS[job_id].update({
                    "status": "completed",
                    "bytes_total": total, "bytes_done": total,
                    "path": local_path,
                })
    except Exception as e:  # noqa: BLE001
        log.exception("download failed")
        try:
            db.mark_hf_download_failed(job_id, str(e))
        except Exception:  # noqa: BLE001
            pass
        if job_id in _DOWNLOADS:
            _DOWNLOADS[job_id].update({"status": "error", "error": str(e)})


def restore_in_progress_downloads() -> int:
    """Re-populate the in-memory _DOWNLOADS dict from the DB on startup.

    Jobs that were 'downloading' when the service died get flipped to
    'cancelled' (we can't resume a partial download — the child process
    is gone). 'queued' jobs are restored as-is so a queued download
    started by the previous session can be observed.

    Returns the count of restored rows.
    """
    from finetune_studio import db
    in_progress = db.list_hf_downloads_in_progress(limit=200)
    restored = 0
    for row in in_progress:
        job_id = row["id"]
        if job_id in _DOWNLOADS:
            continue
        # A status='downloading' from a dead process is no longer valid.
        status = row["status"]
        if status == "downloading":
            db.mark_hf_download_failed(job_id, "service restarted while downloading")
            status = "error"
            row["status"] = status
            row["error"] = "service restarted while downloading"
        _DOWNLOADS[job_id] = {
            "job_id": job_id,
            "status": status,
            "repo_id": row.get("repo_id", ""),
            "filename": row.get("filename", ""),
            "started_at": row.get("started_at"),
            "finished_at": row.get("finished_at"),
            "created_at": row.get("created_at"),
            "bytes_done": row.get("bytes_done") or 0,
            "bytes_total": row.get("bytes_total"),
            "path": row.get("path"),
            "error": row.get("error"),
        }
        restored += 1
    if restored:
        log.info("Restored %d HF download job(s) from DB", restored)
    return restored


@router.get("/hf/local")
async def hf_local():
    """List models already downloaded into our local cache."""
    return {"models": _local_models()}


@router.delete("/hf/local/{repo_id:path}")
async def hf_delete_local(repo_id: str):
    """Delete a locally cached model."""
    target = _LOCAL / repo_id.replace("/", "__")
    if not target.exists():
        return JSONResponse({"error": "not found"}, status_code=404)
    shutil.rmtree(target)
    return {"ok": True, "deleted": repo_id}


@router.get("/hf/local/{repo_id:path}/files")
async def hf_local_files(repo_id: str):
    """List files in a local model dir with sizes."""
    target = _LOCAL / repo_id.replace("/", "__")
    if not target.exists():
        return JSONResponse({"error": "not found"}, status_code=404)
    files = []
    for p in sorted(target.rglob("*")):
        if p.is_file():
            files.append({
                "path": str(p.relative_to(target)),
                "size_bytes": p.stat().st_size,
                "modified": p.stat().st_mtime,
            })
    return {"repo_id": repo_id, "path": str(target), "files": files, "count": len(files)}


# Shared model library (also serves as the model-pool for RAG embedders)
@router.get("/shared-models/stats")
async def shared_model_stats_endpoint():
    from finetune_studio.data.shared_models import stats as sm_stats
    return sm_stats()


# ── Model favorites ──────────────────────────────────────────────
@router.get("/favorites")
async def list_favorites():
    """List favorited models."""
    from finetune_studio import db
    return db.list_model_favorites()

@router.post("/favorites")
async def add_favorite(request: Request):
    """Add a model to favorites."""
    from finetune_studio import db
    body = await request.json() if request.headers.get('content-type') == 'application/json' else {}
    path = body.get('path', '')
    name = body.get('name', path.split('/')[-1] if path else '')
    note = body.get('note', '')
    if not path:
        return JSONResponse({"error": "path required"}, status_code=400)
    db.add_model_favorite(model_path=path, name=name, note=note)
    return {"ok": True}

@router.delete("/favorites/{path:path}")
async def remove_favorite(path: str):
    """Remove a model from favorites."""
    from finetune_studio import db
    db.remove_model_favorite(path)
    return {"ok": True}
