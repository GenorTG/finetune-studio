"""Routes for browsing + downloading HuggingFace models.

UI: LM Studio-style model explorer. Search the HF Hub, filter by task,
inspect model cards, download a specific file or full repo to a local cache.

Local cache: ~/.finetune-studio/hf_models/<repo_id>/<file>  (NOT the default
HF hub cache, because we want a single user-facing directory listing).
On disk ``/`` in the repo id becomes ``__``; ``GET /hf/local`` therefore
reports that mangled directory name as ``repo_id``.
"""

from __future__ import annotations

import asyncio
import logging
import re
from pathlib import Path

from fastapi import APIRouter, HTTPException, Query, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel

log = logging.getLogger(__name__)
router = APIRouter()

_LOCAL = Path.home() / ".finetune-studio" / "hf_models"
_REPO_SEGMENT = re.compile(r"^[A-Za-z0-9._-]+$")


def _local_dir(repo_id: str) -> Path:
    """Cache dir for ``repo_id``; 400 on anything that could escape ``_LOCAL``.

    Every ``/``-separated segment must be ``[A-Za-z0-9._-]+`` and not ``.`` or
    ``..`` — otherwise ``DELETE /hf/local/..`` would ``rmtree`` the parent.
    """
    segments = repo_id.split("/")
    if not all(
        _REPO_SEGMENT.match(seg) and seg not in (".", "..") for seg in segments
    ):
        raise HTTPException(status_code=400, detail="invalid repo_id")
    return _LOCAL / repo_id.replace("/", "__")


# ── DTOs ────────────────────────────────────────────────────────────────

class SearchRequest(BaseModel):
    query: str = ""
    task: str | None = "text-generation"  # text-generation, image-text-to-text, ...
    library: str | None = None           # transformers, sentence-transformers, ...
    sort: str = "downloads"                  # downloads | likes | trending
    limit: int = 24


# ── Local model library ────────────────────────────────────────────────

def _local_models() -> list[dict]:
    """Walk the local cache and return one entry per (repo_id, branch_or_file)."""
    _LOCAL.mkdir(parents=True, exist_ok=True)
    from finetune_studio import db

    identities: dict[str, set[str]] = {}
    for job in db.list_hf_downloads_recent(limit=5000):
        repo_id = str(job.get("repo_id") or "")
        if repo_id:
            identities.setdefault(repo_id.replace("/", "__"), set()).add(repo_id)
    active_repos = {
        str(job.get("repo_id") or "")
        for job in db.list_hf_downloads_in_progress(limit=10000)
    }
    out = []
    for repo_dir in sorted(_LOCAL.iterdir()):
        if not repo_dir.is_dir():
            continue
        # Each repo can have a "snapshots" subdir (HF cache layout) or just files.
        files = []
        size = 0
        for p in sorted(repo_dir.rglob("*")):
            if p.is_file():
                n = p.stat().st_size
                size += n
                files.append({
                    "path": str(p.relative_to(_LOCAL)),
                    "size_bytes": n,
                })
        known_ids = identities.get(repo_dir.name, set())
        unambiguous_repo_id = next(iter(known_ids)) if len(known_ids) == 1 else None
        out.append({
            # Keep the legacy field stable; the folder name is a lossy encoding.
            # New callers should use hub_repo_id, which exists only when durable
            # download history identifies the original id unambiguously.
            "repo_id": repo_dir.name,
            "hub_repo_id": unambiguous_repo_id,
            "repo_id_known": unambiguous_repo_id is not None,
            "downloading": unambiguous_repo_id in active_repos if unambiguous_repo_id else False,
            "cache_key": repo_dir.name,
            "path": str(repo_dir),
            "size_bytes": size,
            "file_count": len(files),
            "files": files[:50],  # cap listing
        })
    return out


# ── Hub search ─────────────────────────────────────────────────────────

class HFUnavailableError(RuntimeError):
    """HuggingFace Hub could not be reached or returned an error."""


# Hub >=1.0 omits lastModified from list_models unless it is expanded explicitly.
_LIST_EXPAND = ["downloads", "likes", "tags", "lastModified", "private"]


def _iso_timestamp(value: object) -> str | None:
    """Hub timestamps arrive as datetime (hub 1.x) or str; absent -> None, never the string 'None'."""
    if value is None or value == "":
        return None
    isoformat = getattr(value, "isoformat", None)
    return isoformat() if callable(isoformat) else str(value)


def _search_hf(req: SearchRequest) -> list[dict]:
    """Query the HuggingFace Hub.

    Two surfaced bugs were fixed here:
    1. Free-text queries like ``"qwen3 gguf"`` previously required the literal
       string to be a substring of the repo_id, so they almost always returned
       zero results even though the Hub search matched plenty of models.
       We now tokenise the query and require every token to be present in the
       repo_id (case-insensitive).
    2. The ``pipeline_tag`` filter (``text-generation``) silently excluded
       models tagged only ``conversational``, which is how Qwen3 chat models
       are commonly published. When the strict filter yields zero results we
       retry once without it so the user still sees the matches.
    """
    from huggingface_hub import HfApi
    api = HfApi()
    sort = {"downloads": "downloads", "likes": "likes",
            "trending": "trendingScore"}.get(req.sort, "downloads")
    query_tokens = [t for t in re.split(r"\s+", (req.query or "").lower().strip()) if t]
    def _matches(mid: str) -> bool:
        if not query_tokens:
            return True
        lowered = mid.lower()
        return all(tok in lowered for tok in query_tokens)
    def _list_models(pipeline_tag: str | None) -> list[dict]:
        out: list[dict] = []
        # list_models is lazy: network errors surface during iteration, so the
        # whole loop (not just the call) must be inside the try.
        try:
            models = api.list_models(
                search=req.query or None,
                pipeline_tag=pipeline_tag or None,
                filter=req.library or None,  # hub >=1.0 dropped library=; a tag filter is the replacement
                sort=sort,
                limit=req.limit * 4 + 20,
                expand=_LIST_EXPAND,
            )
            for m in models:
                mid = getattr(m, "id", None) or m.modelId
                if not _matches(mid):
                    continue
                out.append({
                    "repo_id": mid,
                    "downloads": getattr(m, "downloads", 0) or 0,
                    "likes": getattr(m, "likes", 0) or 0,
                    "tags": getattr(m, "tags", []) or [],
                    "last_modified": _iso_timestamp(
                        getattr(m, "last_modified", None) or getattr(m, "lastModified", None)),
                    "private": getattr(m, "private", False),
                })
                if len(out) >= req.limit:
                    break
        except (TypeError, AttributeError):
            raise  # a call-signature/attribute bug is ours, not a Hub outage
        except Exception as e:
            log.warning("HF list_models failed: %s", e)
            raise HFUnavailableError(str(e)) from e
        return out
    out = _list_models(req.task or None)
    if not out and req.task:
        # Pipeline-tag was too strict — retry without it.
        out = _list_models(None)
    return out


def _model_info(repo_id: str) -> dict | None:
    from huggingface_hub import HfApi
    api = HfApi()
    try:
        info = api.model_info(repo_id)
        files = []
        try:
            siblings = api.list_repo_files(repo_id, repo_type="model")
        except Exception:  # noqa: BLE001
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
            "last_modified": _iso_timestamp(
                getattr(info, "last_modified", None) or getattr(info, "lastModified", None)),
            "card_data": {
                "description": (getattr(info, "card_data", {}) or {}).get("description", ""),
                "license": (getattr(info, "card_data", {}) or {}).get("license", ""),
                "tags": (getattr(info, "card_data", {}) or {}).get("tags", []),
            } if getattr(info, "card_data", None) else {},
            "files": files,
        }
    except Exception as e:  # noqa: BLE001
        log.warning("HF model_info failed: %s", e)
        return None


# ── Routes ──────────────────────────────────────────────────────────────

@router.get("/hf/search")
async def hf_search(q: str = "", task: str = "text-generation",
                   library: str | None = None,
                   sort: str = "downloads", limit: int = Query(24, ge=1, le=100)):
    req = SearchRequest(query=q, task=task, library=library, sort=sort, limit=limit)
    try:
        results = _search_hf(req)
    except HFUnavailableError as e:
        return JSONResponse(
            {"detail": f"Hugging Face Hub is unreachable right now: check your connection and retry ({str(e)[:200]})",
             "code": "hub_unreachable", "results": [], "task": task, "query": q},
            status_code=502,
        )
    return {"results": results, "task": task, "query": q}


@router.get("/hf/info/{repo_id:path}")
async def hf_info(repo_id: str):
    info = _model_info(repo_id)
    if info is None:
        return JSONResponse({"detail": "model not found or fetch failed"}, status_code=404)
    return info


class DownloadRequest(BaseModel):
    repo_id: str
    filename: str | None = None  # if set, download just that file; else full repo
    revision: str = "main"


@router.post("/hf/download")
async def hf_download(req: DownloadRequest):
    """Start a download as its own job process (``webui.download_jobs``); return at once with its job_id.
    Track progress via /hf/download/progress?job_id=... A web restart does not end the download."""
    from finetune_studio.jobs import launch
    from finetune_studio.webui import download_jobs
    from finetune_studio.webui.errors import ApiError

    dest = _local_dir(req.repo_id)  # reject traversal before queueing
    try:
        row = await asyncio.to_thread(download_jobs.start, repo_id=req.repo_id, filename=req.filename,
                                      revision=req.revision, dest=str(dest))
    except launch.JobRejected as e:
        raise ApiError(e.status if e.status >= 400 else 503, f"could not start the download: {e}",
                       code=e.code or None) from e
    return {"ok": True, "job_id": row["id"], "started_at": row.get("started_at")}


@router.get("/hf/download/progress")
async def hf_download_progress(job_id: str):
    """The download row (the job process writes it), with live byte counts while it runs."""
    from finetune_studio.webui import download_jobs

    p = await asyncio.to_thread(download_jobs.progress, job_id)
    if p is None:
        return JSONResponse({"detail": "unknown job_id"}, status_code=404)
    return p


@router.post("/hf/download/cancel")
async def hf_download_cancel(job_id: str):
    """Cancel a download: the row says cancelled at once and its job process is stopped."""
    from finetune_studio.webui import download_jobs

    status = await asyncio.to_thread(download_jobs.cancel, job_id)
    if status is None:
        return JSONResponse({"detail": "unknown job_id"}, status_code=404)
    return {"ok": True, "job_id": job_id, "status": status}


def restore_in_progress_downloads() -> int:
    """Web startup: downloads whose job process is gone are marked failed; running ones are left alone.

    Returns the number of rows marked failed (``download_jobs.reconcile_on_startup``)."""
    from finetune_studio.webui import download_jobs

    return download_jobs.reconcile_on_startup()


@router.get("/hf/local")
async def hf_local():
    """List models already downloaded into our local cache."""
    return {"models": _local_models()}


@router.delete("/hf/local/{repo_id:path}")
async def hf_delete_local(repo_id: str):
    """Delete a locally cached model."""
    try:
        from finetune_studio.models.hf_cache import HFCacheError, delete_cached_model

        await asyncio.to_thread(delete_cached_model, repo_id)
    except HFCacheError as exc:
        return JSONResponse({"detail": str(exc)}, status_code=exc.status_code)
    except Exception as exc:
        log.exception("HF model delete failed")
        return JSONResponse({"detail": f"could not delete downloaded model: {exc}"}, status_code=500)
    return {"ok": True, "deleted": repo_id}


@router.get("/hf/local/{repo_id:path}/files")
async def hf_local_files(repo_id: str):
    """List files in a local model dir with sizes."""
    target = _local_dir(repo_id)
    if not target.exists():
        return JSONResponse({"detail": "not found"}, status_code=404)
    def _list() -> list[dict]:
        out = []
        for p in sorted(target.rglob("*")):
            if p.is_file():
                st = p.stat()
                out.append({
                    "path": str(p.relative_to(target)),
                    "size_bytes": st.st_size,
                    "modified": st.st_mtime,
                })
        return out

    files = await asyncio.to_thread(_list)
    return {"repo_id": repo_id, "path": str(target), "files": files, "count": len(files)}


# Shared model library (also serves as the model-pool for RAG embedders)
@router.get("/shared-models/stats")
async def shared_model_stats_endpoint():
    from finetune_studio.data.shared_models import stats as sm_stats
    return sm_stats()


# ── Model favorites ──────────────────────────────────────────────
@router.get("/favorites")
async def list_favorites():
    """List favorited models. Never 500 the page — empty list on error."""
    from finetune_studio import db
    try:
        return db.list_model_favorites()
    except Exception:
        log.exception("list_model_favorites failed")
        return []

@router.post("/favorites")
async def add_favorite(request: Request):
    """Add a model to favorites."""
    from finetune_studio import db
    body: object = {}
    if request.headers.get("content-type", "").startswith("application/json"):
        try:
            body = await request.json()
        except ValueError:
            body = {}
    if not isinstance(body, dict):
        body = {}
    path = body.get('path', '')
    name = body.get('name', path.split('/')[-1] if path else '')
    note = body.get('note', '')
    if not path or not isinstance(path, str):
        return JSONResponse({"detail": "path required"}, status_code=400)
    db.add_model_favorite(model_path=path, name=name, note=note)
    return {"ok": True}

@router.delete("/favorites/{path:path}")
async def remove_favorite(path: str):
    """Remove a model from favorites."""
    from finetune_studio import db
    db.remove_model_favorite(path)
    return {"ok": True}
