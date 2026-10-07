"""Routes for the AI-powered Data Prep workflow + Model provider management.

Uses the structured project filesystem (`finetune_studio.data.project_filesystem`):
  - content-addressed file store (dedup by sha256)
  - per-file parsed.txt / parsed.json / chunks/
  - per-project logs/ingestions.jsonl audit trail
  - Q&A on disk in qa/pairs/<id>.json + qa/sources/<id>.json
"""

from __future__ import annotations

import asyncio
import json
import logging
import time
from pathlib import Path
from typing import Literal

from fastapi import (
    APIRouter,
    BackgroundTasks,
    File,
    Form,
    HTTPException,
    Request,
    UploadFile,
)
from fastapi.responses import HTMLResponse, JSONResponse, Response
from pydantic import BaseModel, Field

from finetune_studio import __release_channel__ as RELEASE_CHANNEL
from finetune_studio import __version__ as APP_VERSION
from finetune_studio.data.fs.paths import resolve_in_project

log = logging.getLogger(__name__)

router = APIRouter()
_pages = APIRouter()


def _project_404(pid: str) -> JSONResponse | None:
    """Return a 404 response when the project does not exist, else None.

    This module answers errors with ``JSONResponse`` (see ``get_prep_run``),
    so the guard matches that style. Called before any project filesystem
    walk, coverage-fill pass or stream so a bad pid starts no work.
    """
    from finetune_studio import db
    if not db.get_project(pid):
        return JSONResponse({"error": "project not found"}, status_code=404)
    return None

# In-memory run registry. Keyed by (pid, run_id).
_RUNS: dict[tuple[str, str], dict] = {}

# Strong refs to fire-and-forget tasks so the event loop cannot GC them mid-run.
_BG_TASKS: set[asyncio.Task] = set()
_QA_EDITABLE_FIELDS = frozenset({"question", "answer", "status", "chunk_idx", "note"})


def _spawn_bg(coro) -> None:
    task = asyncio.get_running_loop().create_task(coro)
    _BG_TASKS.add(task)
    task.add_done_callback(_BG_TASKS.discard)

Difficulty = Literal["easy", "medium", "hard", "expert"]
Style = Literal["socratic", "direct", "factual", "eli5", "code"]


class StartPrepBody(BaseModel):
    source_id: str
    qa_per_chunk: int = Field(default=3, ge=1, le=10)
    difficulty: Difficulty = "medium"
    style: Style = "socratic"


class BulkPrepBody(BaseModel):
    source_ids: list[str] = Field(min_length=1, max_length=500)
    qa_per_chunk: int = Field(default=3, ge=1, le=10)
    difficulty: Difficulty = "medium"
    style: Style = "socratic"


def _progress_cb(progress_log: list[dict]) -> object:
    """Build a PrepProgress callback that appends to ``progress_log``."""

    def _cb(p: object) -> None:
        progress_log.append({
            "stage": p.stage, "pct": p.pct, "message": p.message,  # type: ignore[attr-defined]
            "source_id": p.source_id, "sha256": p.sha256,  # type: ignore[attr-defined]
            "chunks_total": p.chunks_total, "chunks_done": p.chunks_done,  # type: ignore[attr-defined]
            "qa_total": p.qa_total, "ts": time.time(),  # type: ignore[attr-defined]
        })

    return _cb


def _run_prep_background(run_id: str, runner: object, progress_log: list[dict]) -> None:
    """Shared background body for upload / start / reprocess prep runs."""
    from finetune_studio import db

    try:
        db.mark_data_prep_running(run_id)
    except Exception:
        log.exception("mark_data_prep_running failed for %s", run_id)
    try:
        result = runner.run()  # type: ignore[attr-defined]
        progress_log.append({
            "stage": "done" if result.get("ok") else "error",
            "pct": 100 if result.get("ok") else 0,
            "message": json.dumps(result), "ts": time.time(),
        })
        qa_total = 0
        for entry in progress_log:
            if isinstance(entry.get("qa_total"), int):
                qa_total = max(qa_total, entry["qa_total"])
        qa_total = max(
            qa_total,
            int(result.get("qa") or 0) + int(result.get("qa_coverage_fill") or 0),
        )
        if result.get("ok"):
            try:
                db.mark_data_prep_done(
                    run_id, qa_total=qa_total, qa_approved=0,
                    output_path=result.get("output_path", "") or "",
                )
            except Exception:
                log.exception("mark_data_prep_done failed for %s", run_id)
        else:
            try:
                db.mark_data_prep_failed(
                    run_id,
                    str(result.get("error") or result.get("message") or "prep failed"),
                )
            except Exception:
                log.exception("mark_data_prep_failed failed for %s", run_id)
    except Exception as e:
        log.exception("data-prep background task failed")
        progress_log.append({"stage": "error", "pct": 0,
                             "message": str(e), "ts": time.time()})
        try:
            db.mark_data_prep_failed(run_id, str(e))
        except Exception:
            log.exception("mark_data_prep_failed after crash failed for %s", run_id)

def enqueue_prep_run(
    pid: str,
    background: BackgroundTasks,
    *,
    data: bytes,
    filename: str,
    qa_per_chunk: int = 3,
    difficulty: str = "medium",
    style: str = "socratic",
    uploaded_by: str = "webui",
    source_id: str = "",
    settings_obj: dict | None = None,
) -> str:
    """Create a DB run row, register the runner, queue background work.

    Returns the run_id. Used by ``/upload`` (``/start`` uses ``enqueue_source_prep_run``).
    """
    from finetune_studio import db
    from finetune_studio.data.prep import DataPrepRunner

    db_row = db.create_data_prep_run(
        project_id=pid, filename=filename, byte_count=len(data),
        source_id=source_id, settings_obj=settings_obj,
    )
    run_id = db_row["id"]
    progress_log: list[dict] = []
    runner = DataPrepRunner(
        pid=pid, data=data, filename=filename,
        qa_per_chunk=qa_per_chunk, difficulty=difficulty, style=style,
        uploaded_by=uploaded_by, progress_cb=_progress_cb(progress_log),
    )
    _RUNS[(pid, run_id)] = {
        "runner": runner, "log": progress_log,
        "filename": filename, "byte_count": len(data),
    }
    background.add_task(_run_prep_background, run_id, runner, progress_log)
    return run_id


def enqueue_source_prep_run(
    pid: str,
    background: BackgroundTasks,
    *,
    source: dict,
    qa_per_chunk: int,
    difficulty: str,
    style: str,
) -> str:
    """Queue one source without retaining its potentially large bytes in RAM."""
    from finetune_studio import db
    from finetune_studio.data.prep.queued import QueuedSourcePrep

    path_str = source.get("data_path") or source.get("path") or ""
    if not path_str:
        raise FileNotFoundError(str(source.get("id") or "source"))
    path = resolve_in_project(pid, path_str, what="source path")
    if not path.is_file():
        raise FileNotFoundError(path_str)
    filename = source.get("filename") or source.get("name") or path.name
    settings_obj = {
        "source": "per-file-queue",
        "source_id": source.get("id"),
        "qa_per_chunk": qa_per_chunk,
        "difficulty": difficulty,
        "style": style,
    }
    db_row = db.create_data_prep_run(
        project_id=pid,
        filename=filename,
        byte_count=path.stat().st_size,
        source_id=str(source.get("id") or ""),
        settings_obj=settings_obj,
    )
    run_id = db_row["id"]
    progress_log: list[dict] = []
    runner = QueuedSourcePrep(
        pid=pid,
        path=path,
        filename=filename,
        qa_per_chunk=qa_per_chunk,
        difficulty=difficulty,
        style=style,
        uploaded_by="per-file-queue",
        progress_cb=_progress_cb(progress_log),
    )
    _RUNS[(pid, run_id)] = {
        "runner": runner,
        "log": progress_log,
        "filename": filename,
        "byte_count": path.stat().st_size,
    }
    background.add_task(_run_prep_background, run_id, runner, progress_log)
    return run_id


async def resume_stale_data_prep_runs() -> dict[str, int]:
    """Resume queued/running data-prep runs left behind by a process restart.

    A restart (crash, systemd `daemon-reload`, `update.sh`) used to mark
    every in-flight run `failed` outright (`db.reconcile_stale_data_prep`),
    silently dropping the rest of a multi-file batch. Each queued run is
    already durable (project_id, source_id, settings_json, and the source's
    on-disk path via `project_filesystem`), so re-derive the same
    `QueuedSourcePrep` the original request would have built and re-enqueue
    it. Only runs whose source is now missing fall back to `failed`.
    """
    from starlette.concurrency import run_in_threadpool

    from finetune_studio import db
    from finetune_studio.data import project_filesystem as pfs
    from finetune_studio.data.prep.queued import QueuedSourcePrep

    resumed = 0
    failed = 0
    for row in db.list_stale_data_prep_runs():
        rid = row["id"]
        pid = row["project_id"]
        # `db.row_to_dict` already parses the `settings_json` column into a
        # `settings` dict (see `db/connection.py:row_to_dict`) — do not
        # re-parse a "settings_json" key here, it does not exist on this dict.
        settings_obj = row.get("settings") or {}
        source_id = row.get("source_id") or settings_obj.get("source_id") or ""
        source = None
        if source_id and pid:
            try:
                source = next(
                    (s for s in pfs.list_qa_sources(pid) if s.get("id") == source_id),
                    None,
                )
            except Exception:
                log.exception("resume_stale_data_prep_runs: list_qa_sources failed for %s", pid)
        path_str = (source or {}).get("data_path") or (source or {}).get("path") or ""
        try:
            path = resolve_in_project(pid, path_str, what="source path") if path_str else None
        except HTTPException:
            path = None  # legacy out-of-project source: treat as unavailable
        if source is None or path is None or not path.is_file():
            db.mark_data_prep_failed(
                rid, "interrupted by service restart: source file no longer available"
            )
            failed += 1
            continue
        db.update_data_prep_run(rid, status="queued")
        progress_log: list[dict] = []
        filename = row.get("filename") or source.get("filename") or path.name
        runner = QueuedSourcePrep(
            pid=pid,
            path=path,
            filename=filename,
            qa_per_chunk=int(settings_obj.get("qa_per_chunk", 3)),
            difficulty=settings_obj.get("difficulty", "medium"),
            style=settings_obj.get("style", "socratic"),
            uploaded_by="resume-after-restart",
            progress_cb=_progress_cb(progress_log),
        )
        _RUNS[(pid, rid)] = {
            "runner": runner, "log": progress_log,
            "filename": filename, "byte_count": row.get("byte_count") or 0,
        }
        _spawn_bg(run_in_threadpool(_run_prep_background, rid, runner, progress_log))
        resumed += 1
    return {"resumed": resumed, "failed": failed}


# ── HTML page ────────────────────────────────────────────────────────────

@_pages.get("/projects/{pid}/data-prep", response_class=HTMLResponse)
async def data_prep_page(request: Request, pid: str):
    from fastapi.templating import Jinja2Templates

    from finetune_studio import db
    from finetune_studio.data import project_filesystem as pfs
    from finetune_studio.webui.app import discovered_models
    templates = Jinja2Templates(directory=str(Path(__file__).parent.parent / "templates"))
    templates.env.globals["app_version"] = APP_VERSION
    templates.env.globals["release_channel"] = RELEASE_CHANNEL
    project = db.get_project(pid)
    if not project:
        return HTMLResponse("Project not found", status_code=404)
    sources = pfs.list_qa_sources(pid)
    from finetune_studio.data.fs.qa import AUTO_PROMOTE_EXTENSIONS
    from finetune_studio.data.parsers import PARSERS
    from finetune_studio.models.helper import (
        DEFAULT_HELPER_LABEL,
        get_configured_helper_provider,
        get_helper_provider_id,
    )
    helper = get_configured_helper_provider()
    ctx = {
        "request": request,
        "pid": pid,
        "project": project,
        "models": discovered_models,
        "sources": sources,
        "helper_label": (helper or {}).get("label") or DEFAULT_HELPER_LABEL,
        "helper_provider_id": get_helper_provider_id(),
        # Text uploads auto-promote; any parser-supported file can be
        # promoted manually with "Use as source" (register_qa_source parses it).
        "auto_promote_extensions": sorted(AUTO_PROMOTE_EXTENSIONS),
        "promotable_extensions": sorted(PARSERS),
    }
    return templates.TemplateResponse(request, "data_prep.html", ctx)


# RAG page lives in pages.project_rag_page (passes indexed_docs).


# ── Provider CRUD ───────────────────────────────────────────────────────

@router.get("/providers")
async def list_providers():
    from finetune_studio.models.helper import (
        DEFAULT_HELPER_LABEL,
        get_configured_helper_provider,
        get_helper_provider_id,
    )
    from finetune_studio.models.manager import get_manager
    mgr = get_manager()
    helper = get_configured_helper_provider()
    return {
        "providers": mgr.list_providers(),
        "active": mgr.active(),
        "helper": helper,
        "helper_provider_id": get_helper_provider_id(),
        "helper_label": (helper or {}).get("label") or DEFAULT_HELPER_LABEL,
    }


@router.post("/providers")
async def upsert_provider(request: Request):
    body = await request.json()
    from finetune_studio.models.manager import get_manager
    return get_manager().upsert_provider(**body)


@router.delete("/providers/{pid}")
async def delete_provider(pid: str):
    from finetune_studio.models.manager import get_manager
    return {"ok": get_manager().delete_provider(pid)}


@router.post("/providers/{pid}/load")
async def load_provider(pid: str, request: Request):
    """Load a model provider. Accepts an optional JSON body of loader
    params (n_ctx, n_gpu_layers, n_batch, n_threads, seed,
    rope_freq_base, rope_freq_scale, flash_attn, mmap, mlock) which are
    merged into the provider's persisted extras for this load. Pass
    through to ModelManager.load(pid, extra=...)."""
    from finetune_studio.models.llama_loader import resolve_loader_overrides
    from finetune_studio.models.manager import get_manager
    extra = {}
    try:
        body = await request.json()
        if isinstance(body, dict):
            extra = body.get("extra") or body
    except (json.JSONDecodeError, ValueError, TypeError) as e:
        # No body / empty body / non-JSON -> just reload with persisted extras.
        log.debug("load_provider body ignored: %s", e)
    # default_ctx=False: an unset n_ctx must fall through to the provider's
    # persisted value (ModelManager.load's own merge), not get overwritten
    # with the generic 32k floor on every single load.
    extra = resolve_loader_overrides(extra, caller="providers/load", model_path=pid, default_ctx=False)
    from finetune_studio.models.helper import (
        helper_missing_message,
        missing_gguf_for_provider,
    )
    missing = missing_gguf_for_provider(pid)
    if missing:
        return JSONResponse(
            {"ok": False, "code": "helper_missing", "missing_path": missing,
             "error": helper_missing_message(missing)},
            status_code=400,
        )
    # Free whatever the global inference_engine (testing/chat/RAG/benchmarks)
    # has resident before loading this provider — otherwise both sit in
    # VRAM simultaneously until someone happens to click Unload.
    # Both the unload and the load below take a minute for a 12B model. They used to run inline in this
    # async handler, which froze the whole event loop (every page, status poll and SSE stream) for the length
    # of the load — the helper load that starts every mining run. Off the loop, serialised like /models/load.
    from finetune_studio.webui.engine_guard import ENGINE_LOCK
    async with ENGINE_LOCK:
        try:
            from finetune_studio.webui.app import inference_engine
            if getattr(inference_engine, "model", None) is not None:
                await asyncio.to_thread(inference_engine.unload)
        except Exception:
            log.exception("providers/load: failed to unload the global inference engine first")
        try:
            active = await asyncio.to_thread(get_manager().load, pid, extra=extra)
            return {"ok": True, "active": active}
        except Exception as e:
            log.exception("load failed")
            return JSONResponse({"ok": False, "error": str(e)}, status_code=400)


@router.get("/providers/helper/status")
async def helper_status():
    """Is the helper GGUF present, and which installed GGUFs could replace it."""
    from finetune_studio.models.helper import (
        get_configured_helper_provider,
        get_helper_provider_id,
        helper_missing_message,
        missing_gguf_for_provider,
    )
    from finetune_studio.webui.app import discovered_models
    helper = get_configured_helper_provider() or {}
    missing = missing_gguf_for_provider(get_helper_provider_id())
    candidates = [
        {"path": m.path, "name": m.name, "size_gb": m.size_gb}
        for m in discovered_models
        if str(getattr(m, "format", "")).lower() == "gguf"
    ]
    return {
        "ok": not missing,
        "path": helper.get("model_id") or "",
        "missing_path": missing,
        "message": helper_missing_message(missing) if missing else "",
        "candidates": candidates,
    }


@router.post("/providers/helper/use")
async def helper_use(request: Request):
    """Point the helper seat (``local-default``) at an installed GGUF file."""
    from finetune_studio.models.helper import DEFAULT_HELPER_PROVIDER_ID
    from finetune_studio.models.manager import get_manager
    body = await request.json()
    path = str((body or {}).get("path") or "").strip()
    resolved = Path(path).expanduser()
    if not path or resolved.suffix.lower() != ".gguf" or not resolved.is_file():
        return JSONResponse({"ok": False, "error": f"Not a GGUF file: {path or '(empty)'}"}, status_code=400)
    get_manager().upsert_provider(id=DEFAULT_HELPER_PROVIDER_ID, model_id=str(resolved))
    return {"ok": True, "path": str(resolved)}


@router.post("/providers/unload")
async def unload_active():
    """Unload the active provider (both engines — see E2E-22)."""
    from finetune_studio.data.rag_portable.model_cache import release_rag_models
    from finetune_studio.models.llama_loader import unload_all_models
    await asyncio.to_thread(unload_all_models)  # seconds for a resident 12B: never on the event loop
    await asyncio.to_thread(release_rag_models, "unload all models")
    return {"ok": True, "active": None}


# ── Data Prep: upload, process, curate, export ──────────────────────────

@router.post("/projects/{pid}/data-prep/upload", response_model=None)
async def upload_file(
    pid: str,
    background: BackgroundTasks,
    file: UploadFile = File(...),  # noqa: B008  # FastAPI requires File() default at def site
    qa_per_chunk: int = Form(3),
    difficulty: str = Form("medium"),
    style: str = Form("socratic"),
):
    """Read bytes, then start a prep run in background. NO auto-load of model."""
    data = await file.read()
    if not data:
        return JSONResponse({"error": "empty upload"}, status_code=400)
    filename = file.filename or "upload"
    run_id = enqueue_prep_run(
        pid, background,
        data=data, filename=filename,
        qa_per_chunk=qa_per_chunk, difficulty=difficulty, style=style,
        uploaded_by="webui",
    )
    return {"ok": True, "run_id": run_id, "filename": filename, "byte_count": len(data)}


@router.post("/projects/{pid}/data-prep/start", response_model=None)
async def start_prep(
    pid: str,
    body: StartPrepBody,
    background: BackgroundTasks,
):
    """Start prep from an existing QA source (source picker → Start prep)."""
    from finetune_studio.data import project_filesystem as pfs
    from finetune_studio.data.prep.generator import resolve_generator

    sources = pfs.list_qa_sources(pid)
    src = next((s for s in sources if s.get("id") == body.source_id), None)
    if src is None:
        return JSONResponse(
            {"error": f"unknown source_id: {body.source_id}"},
            status_code=404,
        )
    path_str = src.get("data_path") or src.get("path") or ""
    try:
        path = resolve_in_project(pid, path_str, what="source path")
    except HTTPException as exc:
        return JSONResponse({"error": exc.detail}, status_code=exc.status_code)
    if not path_str or not path.is_file():
        return JSONResponse(
            {"error": f"source file missing: {path_str or body.source_id}"},
            status_code=404,
        )
    if resolve_generator() is None:
        from finetune_studio.data.prep.generator import helper_resolution_error

        return JSONResponse({"error": helper_resolution_error()}, status_code=409)

    run_id = enqueue_source_prep_run(
        pid,
        background,
        source=src,
        qa_per_chunk=body.qa_per_chunk,
        difficulty=body.difficulty,
        style=body.style,
    )
    return {"ok": True, "run_id": run_id, "source_id": body.source_id}


@router.post("/projects/{pid}/data-prep/start-bulk", response_model=None)
async def start_bulk_prep(
    pid: str,
    body: BulkPrepBody,
    background: BackgroundTasks,
):
    """Queue selected parsed files as sequential, source-scoped prep jobs.

    Starlette executes these background tasks in insertion order. That is
    intentional: one helper model consumes the available GPU at a time, while
    queued files retain only their paths rather than every file's bytes.
    """
    missing = _project_404(pid)
    if missing is not None:
        return missing
    from finetune_studio.data import project_filesystem as pfs
    from finetune_studio.data.prep.generator import (
        helper_resolution_error,
        resolve_generator,
    )

    if resolve_generator() is None:
        return JSONResponse({"error": helper_resolution_error()}, status_code=409)
    by_id = {str(s.get("id") or ""): s for s in pfs.list_qa_sources(pid)}
    unknown = [source_id for source_id in body.source_ids if source_id not in by_id]
    if unknown:
        return JSONResponse(
            {"error": f"unknown source ids ({len(unknown)}): {unknown[:10]}"},
            status_code=400,
        )
    run_ids: list[str] = []
    for source_id in dict.fromkeys(body.source_ids):
        source = by_id[source_id]
        if source.get("status") in {"queued", "parsing", "error", "registered"}:
            return JSONResponse(
                {"error": f"source {source_id} is not parsed and ready"},
                status_code=409,
            )
        try:
            run_ids.append(
                enqueue_source_prep_run(
                    pid,
                    background,
                    source=source,
                    qa_per_chunk=body.qa_per_chunk,
                    difficulty=body.difficulty,
                    style=body.style,
                )
            )
        except FileNotFoundError as exc:
            return JSONResponse({"error": f"source file missing: {exc}"}, status_code=404)
    return {"ok": True, "run_ids": run_ids, "queued": len(run_ids)}


@router.get("/projects/{pid}/data-prep/runs/{run_id}/events")
async def stream_events(pid: str, run_id: str):
    missing = _project_404(pid)
    if missing is not None:
        return missing
    from fastapi.responses import StreamingResponse
    async def gen():
        run = _RUNS.get((pid, run_id))
        if not run:
            yield "data: " + json.dumps({"stage": "error", "message": "unknown run"}) + "\n\n"
            return
        last, log_list, runner = 0, run["log"], run["runner"]
        while True:
            finished = runner.progress.stage in ("done", "error")
            while last < len(log_list):
                yield "data: " + json.dumps(log_list[last]) + "\n\n"
                last += 1
            if finished:
                return
            await asyncio.sleep(0.5)
    return StreamingResponse(gen(), media_type="text/event-stream",
                             headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})


@router.get("/projects/{pid}/data-prep/runs/{run_id}")
async def get_prep_run(pid: str, run_id: str):
    """One-shot prep-run progress (SSE silent fallback for /events)."""
    run = _RUNS.get((pid, run_id))
    if not run:
        from finetune_studio import db
        row = db.get_data_prep_run(run_id)
        if not row or row.get("project_id") != pid:
            return JSONResponse(
                {"error": "unknown run", "stage": "error"}, status_code=404,
            )
        status = str(row.get("status") or "unknown")
        stage = "done" if status in ("done", "completed") else (
            "error" if status in ("error", "failed") else status
        )
        return {
            "run_id": run_id,
            "stage": stage,
            "pct": 100 if stage == "done" else 0,
            "qa_total": row.get("qa_count") or row.get("pair_count") or 0,
            "message": row.get("error") or "",
            "filename": row.get("filename") or "",
        }
    runner = run["runner"]
    prog = runner.progress
    log_list = run["log"]
    last = log_list[-1] if log_list else {}
    return {
        "run_id": run_id,
        "stage": prog.stage or last.get("stage") or "running",
        "pct": prog.pct if prog.pct is not None else last.get("pct"),
        "qa_total": prog.qa_total if prog.qa_total is not None else last.get("qa_total"),
        "message": last.get("message") or "",
        "filename": run.get("filename") or "",
    }


@router.get("/projects/{pid}/data-prep/sources")
async def list_sources_route(pid: str):
    missing = _project_404(pid)
    if missing is not None:
        return missing
    from finetune_studio.data import project_filesystem as pfs
    sources = pfs.list_qa_sources(pid)
    from finetune_studio.data.prep.source_state import summarize_source

    out = [{
        "id": s["id"], "filename": s["filename"], "mime_type": s.get("mime_type", ""),
        "char_count": s.get("char_count", 0), "chunk_count": s.get("chunk_count", 0),
        "uploaded_at": s.get("uploaded_at", 0), "sha256": s.get("sha256", ""),
        "parser": s.get("parser", ""), "status": s.get("status", "registered"),
        "error": s.get("error", ""),
        "data_path": s.get("data_path") or s.get("path") or "",
        **summarize_source(pid, s),
    } for s in sources]
    return {"sources": out}


@router.post("/projects/{pid}/data-prep/sources")
async def promote_source_route(pid: str, request: Request):
    """Promote a file-library upload into the data-prep source picker (QABUG-003).

    Body accepts either ``file_id`` (resolved via ``project_files`` +
    ``file_versions``) or ``data_path`` (absolute path on disk).
    Idempotent on (pid, path): re-posting returns the existing source row.
    """
    from finetune_studio import db
    from finetune_studio.data import project_filesystem as pfs
    from finetune_studio.data.fs.qa import promote_file_library_upload

    body = await request.json()
    data_path = body.get("data_path") or ""
    mime_type = body.get("mime_type") or ""
    filename = body.get("filename") or None
    fid = body.get("file_id")
    if fid and not data_path:
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
            try:
                source = promote_file_library_upload(
                    pid, fid, mime_type=mime_type, filename=filename
                )
            except (OSError, ValueError) as e:
                log.exception("promote source failed")
                return JSONResponse({"error": str(e)}, status_code=400)
            return {"ok": True, "source": source}
    if not data_path:
        return JSONResponse(
            {"error": "data_path or file_id required"}, status_code=400
        )
    # Project-dir fence: never ingest a file outside the project directory.
    data_path = str(resolve_in_project(pid, data_path, what="data_path"))
    try:
        source = pfs.register_qa_source(
            pid, data_path, mime_type=mime_type, filename=filename
        )
    except OSError as e:
        log.exception("promote source failed")
        return JSONResponse({"error": str(e)}, status_code=400)
    return {"ok": True, "source": source}


@router.get("/projects/{pid}/data-prep/qa")
async def list_qa_route(pid: str, source_id: str | None = None, status: str | None = None):
    missing = _project_404(pid)
    if missing is not None:
        return missing
    from finetune_studio.data import project_filesystem as pfs
    return {"items": pfs.list_qa_pairs(pid, source_id=source_id, status=status)}


@router.post("/projects/{pid}/data-prep/dedupe")
async def dedupe_pairs_route(pid: str, request: Request):
    """Reject pending pairs that restate a fact already asked (same answer tokens, near-identical question).

    Body: ``{"dry_run": bool, "source_id": str|null}``. Only PENDING pairs are touched; the first pair of each cluster stays.
    """
    missing = _project_404(pid)
    if missing is not None:
        return missing
    try:
        body = await request.json()
    except ValueError:
        body = {}
    body = body if isinstance(body, dict) else {}
    from finetune_studio.data import project_filesystem as pfs
    from finetune_studio.data.prep.dedupe import find_near_duplicates

    pairs = await asyncio.to_thread(pfs.list_qa_pairs, pid, source_id=body.get("source_id") or None, status="pending")
    pairs.sort(key=lambda p: (str(p.get("source_id")), int(p.get("chunk_idx") or 0), float(p.get("created_at") or 0), str(p.get("id"))))
    dupes = find_near_duplicates(pairs)
    if not body.get("dry_run"):
        def _apply() -> None:
            for dup_id, kept_id in dupes:
                pfs.update_qa_pair(pid, dup_id, status="rejected", note=f"near-duplicate of {kept_id}")
        await asyncio.to_thread(_apply)
    return {"ok": True, "dry_run": bool(body.get("dry_run")), "rejected": len(dupes), "of_pending": len(pairs),
            "examples": [{"duplicate": d, "kept": k} for d, k in dupes[:10]]}


@router.patch("/projects/{pid}/data-prep/qa/{qa_id}")
async def update_qa_route(pid: str, qa_id: str, request: Request):
    missing = _project_404(pid)
    if missing is not None:
        return missing
    body = await request.json()
    if not isinstance(body, dict):
        return JSONResponse({"error": "JSON object body required"}, status_code=400)
    fields = {k: v for k, v in body.items() if k in _QA_EDITABLE_FIELDS}
    if not fields:
        return JSONResponse(
            {"error": f"no editable fields; allowed: {sorted(_QA_EDITABLE_FIELDS)}"},
            status_code=400,
        )
    from finetune_studio.data import project_filesystem as pfs
    updated = pfs.update_qa_pair(pid, qa_id, **fields)
    if updated is None:
        return JSONResponse({"error": "not found"}, status_code=404)
    return updated


@router.post("/projects/{pid}/data-prep/qa/bulk")
async def bulk_action(pid: str, request: Request):
    missing = _project_404(pid)
    if missing is not None:
        return missing
    body = await request.json()
    ids, action = body.get("ids", []), body.get("action", "")
    new_status = {"approve": "approved", "reject": "rejected"}.get(action)
    if not new_status:
        return JSONResponse({"error": f"unknown action: {action}"}, status_code=400)
    from finetune_studio.data import project_filesystem as pfs
    for qa_id in ids:
        pfs.update_qa_pair(pid, qa_id, status=new_status)
    return {"ok": True, "updated": len(ids)}


@router.delete("/projects/{pid}/data-prep/source/{source_id}")
async def delete_source_route(pid: str, source_id: str):
    missing = _project_404(pid)
    if missing is not None:
        return missing
    from finetune_studio.data import project_filesystem as pfs
    return {"ok": pfs.delete_qa_source(pid, source_id)}


@router.get("/projects/{pid}/data-prep/export")
async def export_qa(pid: str, fmt: str = "sharegpt", only: str = "approved",
                     force: bool = False, grounded_share: float | None = None,
                     distractors: int = 0):
    """Export pairs as JSONL + register the dataset.

    ``grounded_share`` (0-1): fraction of rows rewritten to carry the RAG-chat
    prompt + CONTEXT from the pair's own source chunk. Omitted = auto (40% when
    the project has a built RAG corpus, else off); ``0`` = plain rows only.
    ``distractors`` (0-2): extra other-file chunks in that CONTEXT.
    """
    missing = _project_404(pid)
    if missing is not None:
        return missing
    if only not in ("approved", "pending", "rejected", "all"):
        return JSONResponse({"error": f"unknown only filter: {only}"}, status_code=400)
    from finetune_studio.data.prep.dataset_build import (
        CoverageCheckFailed,
        ExportBlocked,
        coverage_gate,
        persist_export,
    )
    from finetune_studio.data.prep.export import build_qa_export
    from finetune_studio.data.prep.grounding import resolve_grounding
    try:
        grounding = resolve_grounding(pid, grounded_share, distractors)
    except ValueError as e:
        return JSONResponse({"error": str(e)}, status_code=400)
    # 100%-coverage gate (data.prep.dataset_build): never export a dataset with
    # silent coverage holes.
    try:
        coverage_gate(pid, force=force)
    except ExportBlocked as blocked:
        return JSONResponse(
            {
                "error": str(blocked),
                "uncovered_files": blocked.files,
                "uncovered_chunks": blocked.uncovered_chunks[:50],
                "uncovered_count": len(blocked.uncovered_chunks),
                "hint": "pass ?force=true to export anyway with those chunks missing",
            },
            status_code=409,
        )
    except CoverageCheckFailed as exc:
        return JSONResponse({"error": str(exc)}, status_code=500)
    try:
        result = build_qa_export(pid, fmt, only, grounding=grounding)
    except ValueError as e:
        return JSONResponse({"error": str(e)}, status_code=400)
    body = result.body
    n_grounded = result.grounding.grounded if result.grounding else 0
    # The export lives in a stream buffer; persist it to the project's datasets
    # dir so it's selectable from the Training tab and referenceable forever.
    try:
        persist_export(pid, fmt, only, body, n_grounded)
    except Exception:
        # Don't fail the export if the registry step fails — payload still ships.
        log.exception("dataset registry failed")
    return Response(
        body.encode("utf-8"),
        media_type="application/x-ndjson",
        headers={
            "Content-Disposition": f'attachment; filename="{pid}-{fmt}-{only}.jsonl"',
            "X-Rows": str(result.rows),
            "X-Grounded-Rows": str(n_grounded),
        },
    )


# ── Ingestion log, audit + reprocess ───────────────────────────────────

@router.get("/projects/{pid}/data-prep/ingestion-log")
async def ingestion_log_route(pid: str, limit: int = 200):
    missing = _project_404(pid)
    if missing is not None:
        return missing
    from finetune_studio.data import project_filesystem as pfs
    return {"events": pfs.read_ingestion_log(pid, limit=limit)}


@router.get("/projects/{pid}/data-prep/audit")
async def data_prep_audit(pid: str) -> dict:
    """Return deterministic raw-file and curated-dataset fidelity evidence."""
    missing = _project_404(pid)
    if missing is not None:
        return missing
    from finetune_studio.data.audit import audit_project_sources, audit_qa_pairs

    return {"sources": audit_project_sources(pid), "dataset": audit_qa_pairs(pid)}


@router.post("/projects/{pid}/data-prep/reprocess/{source_id}")
async def reprocess_source(pid: str, source_id: str):
    """Re-run the parser + Q&A generation for an existing source."""
    missing = _project_404(pid)
    if missing is not None:
        return missing
    from finetune_studio.data import project_filesystem as pfs
    src = pfs.read_qa_source(pid, source_id)
    if not src:
        return JSONResponse({"error": "source not found"}, status_code=404)
    sha = src.get("sha256")
    if not sha:
        return JSONResponse({"error": "source has no sha256; cannot re-read bytes"}, status_code=400)
    fd = pfs.file_dir(pid, sha)
    # Find the original file inside the sha dir (any name, any extension)
    candidates = [p for p in fd.iterdir() if p.is_file() and p.suffix.lower()]
    # Prefer the file matching src['filename'] (basename only — a ".." in the
    # manifest name must not escape the sha dir); otherwise any file that
    # isn't metadata/parsed/chunks
    original = fd / Path(src.get("filename") or "").name
    if not original.is_file():
        # Fallback: find any file with a recognized extension (skip the metadata/parsed artifacts)
        skip_names = {"metadata.json", "parsed.txt", "parsed.json"}
        skip_dirs = {"chunks"}
        for p in candidates:
            if p.name in skip_names:
                continue
            if p.parent.name in skip_dirs:
                continue
            original = p
            break
    if not original.is_file():
        return JSONResponse({"error": f"original bytes missing in {fd}"}, status_code=400)
    data = await asyncio.to_thread(original.read_bytes)
    # Persist a DB row so this reprocess shows up in the activity feed.
    # Schedule via asyncio (historical behaviour) rather than BackgroundTasks.
    from finetune_studio import db
    from finetune_studio.data.prep import DataPrepRunner

    db_row = db.create_data_prep_run(
        project_id=pid, filename=original.name, byte_count=len(data),
        source_id=source_id,
        settings_obj={"source": "reprocess", "source_id": source_id},
    )
    run_id = db_row["id"]
    progress_log: list[dict] = []
    runner = DataPrepRunner(
        pid=pid, data=data, filename=original.name,
        uploaded_by="reprocess", progress_cb=_progress_cb(progress_log),
    )
    _RUNS[(pid, run_id)] = {
        "runner": runner, "log": progress_log,
        "filename": original.name, "byte_count": len(data),
    }

    _spawn_bg(asyncio.to_thread(_run_prep_background, run_id, runner, progress_log))
    return {"ok": True, "run_id": run_id, "filename": original.name}
