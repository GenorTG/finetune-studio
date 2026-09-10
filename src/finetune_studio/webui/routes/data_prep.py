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
import uuid
from pathlib import Path
from typing import Optional

from fastapi import APIRouter, BackgroundTasks, Request, UploadFile, File
from fastapi.responses import HTMLResponse, JSONResponse, Response
from pydantic import BaseModel

log = logging.getLogger(__name__)

router = APIRouter()
_pages = APIRouter()

# In-memory run registry. Keyed by (pid, run_id).
_RUNS: dict[tuple[str, str], dict] = {}


# ── HTML page ────────────────────────────────────────────────────────────

@_pages.get("/projects/{pid}/data-prep", response_class=HTMLResponse)
async def data_prep_page(request: Request, pid: str):
    from fastapi.templating import Jinja2Templates
    from finetune_studio import db
    from finetune_studio.webui.app import discovered_models
    templates = Jinja2Templates(directory=str(Path(__file__).parent.parent / "templates"))
    project = db.get_project(pid)
    if not project:
        return HTMLResponse("Project not found", status_code=404)
    ctx = {"request": request, "pid": pid, "project": project, "models": discovered_models}
    return templates.TemplateResponse(request, "data_prep.html", ctx)


@_pages.get("/projects/{pid}/rag", response_class=HTMLResponse)
async def rag_page(request: Request, pid: str):
    from fastapi.templating import Jinja2Templates
    from finetune_studio import db
    templates = Jinja2Templates(directory=str(Path(__file__).parent.parent / "templates"))
    project = db.get_project(pid)
    if not project:
        return HTMLResponse("Project not found", status_code=404)
    return templates.TemplateResponse(request, "rag.html", {"request": request, "pid": pid, "project": project})


# ── Provider CRUD ───────────────────────────────────────────────────────

@router.get("/providers")
async def list_providers():
    from finetune_studio.models.manager import get_manager
    mgr = get_manager()
    return {"providers": mgr.list_providers(), "active": mgr.active()}


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
async def load_provider(pid: str):
    from finetune_studio.models.manager import get_manager
    try:
        return {"ok": True, "active": get_manager().load(pid)}
    except Exception as e:
        log.exception("load failed")
        return JSONResponse({"ok": False, "error": str(e)}, status_code=400)


@router.post("/providers/unload")
async def unload_active():
    from finetune_studio.models.manager import get_manager
    get_manager().unload()
    return {"ok": True, "active": None}


# ── Data Prep: upload, process, curate, export ──────────────────────────

@router.post("/projects/{pid}/data-prep/upload")
async def upload_file(pid: str, background: BackgroundTasks, file: UploadFile = File(...)):
    """Read bytes, then start a prep run in background. NO auto-load of model."""
    data = await file.read()
    if not data:
        return JSONResponse({"error": "empty upload"}, status_code=400)
    filename = file.filename or "upload"
    from finetune_studio.data.prep import DataPrepRunner
    # Persist a DB row so this run shows up in the activity feed / project
    # dashboard even after a restart. Use the DB row id as run_id so the
    # in-memory dict and DB stay aligned (and URL paths stay short).
    from finetune_studio import db
    db_row = db.create_data_prep_run(
        project_id=pid, filename=filename, byte_count=len(data),
    )
    run_id = db_row["id"]
    progress_log: list[dict] = []
    def _cb(p):
        progress_log.append({
            "stage": p.stage, "pct": p.pct, "message": p.message,
            "source_id": p.source_id, "sha256": p.sha256,
            "chunks_total": p.chunks_total, "chunks_done": p.chunks_done,
            "qa_total": p.qa_total, "ts": time.time(),
        })
    runner = DataPrepRunner(
        pid=pid, data=data, filename=filename,
        uploaded_by="webui", progress_cb=_cb,
    )
    _RUNS[(pid, run_id)] = {"runner": runner, "log": progress_log,
                              "filename": filename, "byte_count": len(data)}
    def _bg():
        try:
            db.mark_data_prep_running(run_id)
        except Exception:  # noqa: BLE001
            pass
        try:
            result = runner.run()
            progress_log.append({"stage": "done" if result.get("ok") else "error",
                                 "pct": 100 if result.get("ok") else 0,
                                 "message": json.dumps(result), "ts": time.time()})
            qa_total = 0
            for entry in progress_log:
                if isinstance(entry.get("qa_total"), int):
                    qa_total = max(qa_total, entry["qa_total"])
            if result.get("ok"):
                try:
                    db.mark_data_prep_done(
                        run_id, qa_total=qa_total, qa_approved=0,
                        output_path=result.get("output_path", "") or "",
                    )
                except Exception:  # noqa: BLE001
                    pass
            else:
                try:
                    db.mark_data_prep_failed(
                        run_id,
                        str(result.get("error") or result.get("message") or "prep failed"),
                    )
                except Exception:  # noqa: BLE001
                    pass
        except Exception as e:  # noqa: BLE001
            log.exception("data-prep background task failed")
            progress_log.append({"stage": "error", "pct": 0,
                                 "message": str(e), "ts": time.time()})
            try:
                db.mark_data_prep_failed(run_id, str(e))
            except Exception:  # noqa: BLE001
                pass
    background.add_task(_bg)
    return {"ok": True, "run_id": run_id, "filename": filename, "byte_count": len(data)}


@router.get("/projects/{pid}/data-prep/runs/{run_id}/events")
async def stream_events(pid: str, run_id: str):
    from fastapi.responses import StreamingResponse
    async def gen():
        run = _RUNS.get((pid, run_id))
        if not run:
            yield "data: " + json.dumps({"stage": "error", "message": "unknown run"}) + "\n\n"
            return
        last, log_list, runner = 0, run["log"], run["runner"]
        while True:
            while last < len(log_list):
                yield "data: " + json.dumps(log_list[last]) + "\n\n"
                last += 1
            if runner.progress.stage in ("done", "error"):
                yield "data: " + json.dumps(log_list[-1]) + "\n\n"
                return
            await asyncio.sleep(0.5)
    return StreamingResponse(gen(), media_type="text/event-stream",
                             headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})


@router.get("/projects/{pid}/data-prep/sources")
async def list_sources_route(pid: str):
    from finetune_studio.data import project_filesystem as pfs
    sources = pfs.list_qa_sources(pid)
    out = [{
        "id": s["id"], "filename": s["filename"], "mime_type": s.get("mime_type", ""),
        "char_count": s.get("char_count", 0), "chunk_count": s.get("chunk_count", 0),
        "uploaded_at": s.get("uploaded_at", 0), "sha256": s.get("sha256", ""),
        "parser": s.get("parser", ""),
    } for s in sources]
    return {"sources": out}


@router.get("/projects/{pid}/data-prep/qa")
async def list_qa_route(pid: str, source_id: Optional[str] = None, status: Optional[str] = None):
    from finetune_studio.data import project_filesystem as pfs
    return {"items": pfs.list_qa_pairs(pid, source_id=source_id, status=status)}


@router.patch("/projects/{pid}/data-prep/qa/{qa_id}")
async def update_qa_route(pid: str, qa_id: str, request: Request):
    body = await request.json()
    from finetune_studio.data import project_filesystem as pfs
    updated = pfs.update_qa_pair(pid, qa_id, **body)
    if updated is None:
        return JSONResponse({"error": "not found"}, status_code=404)
    return updated


@router.post("/projects/{pid}/data-prep/qa/bulk")
async def bulk_action(pid: str, request: Request):
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
    from finetune_studio.data import project_filesystem as pfs
    return {"ok": pfs.delete_qa_source(pid, source_id)}


@router.get("/projects/{pid}/data-prep/export")
async def export_qa(pid: str, fmt: str = "sharegpt", only: str = "approved"):
    from finetune_studio.data.prep import export_qa_jsonl
    try:
        body = export_qa_jsonl(pid, fmt=fmt, only=only)
    except ValueError as e:
        return JSONResponse({"error": str(e)}, status_code=400)
    # The export lives in a stream buffer; persist it to the project's datasets
    # dir so it's selectable from the Training tab and referenceable forever.
    try:
        from finetune_studio import db
        from pathlib import Path as _P
        from finetune_studio.db.datasets import datasets_dir, count_qa_pairs
        ds_dir = datasets_dir(pid)
        fname = f"{pid}-{fmt}-{only}.jsonl"
        target = ds_dir / fname
        target.write_text(body, encoding="utf-8")
        existing = db.get_dataset_by_path(pid, str(target))
        if not existing:
            db.create_dataset(
                project_id=pid,
                name=target.stem,
                data_path=str(target),
                source="data-prep-export",
                qa_count=count_qa_pairs(str(target)),
                size_bytes=target.stat().st_size,
            )
    except Exception as e:  # noqa: BLE001
        # Don't fail the export if the registry step fails — payload still ships.
        log.warning("dataset registry failed: %s", e)
    return Response(
        body.encode("utf-8"),
        media_type="application/x-ndjson",
        headers={"Content-Disposition": f'attachment; filename="{pid}-{fmt}-{only}.jsonl"'},
    )


# ── New: file inspection + reprocess ───────────────────────────────────

@router.get("/projects/{pid}/data-prep/file/{sha256}")
async def file_metadata_route(pid: str, sha256: str):
    """Read structured metadata for a content-addressed file."""
    from finetune_studio.data import project_filesystem as pfs
    m = pfs.read_file_metadata(pid, sha256)
    if m is None:
        return JSONResponse({"error": "not found"}, status_code=404)
    return m.to_json()


@router.get("/projects/{pid}/data-prep/ingestion-log")
async def ingestion_log_route(pid: str, limit: int = 200):
    from finetune_studio.data import project_filesystem as pfs
    return {"events": pfs.read_ingestion_log(pid, limit=limit)}


@router.post("/projects/{pid}/data-prep/reprocess/{source_id}")
async def reprocess_source(pid: str, source_id: str):
    """Re-run the parser + Q&A generation for an existing source."""
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
    # Prefer the file matching src['filename']; otherwise any file that isn't metadata/parsed/chunks
    original = fd / src.get("filename", "")
    if not original.exists():
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
    if not original.exists():
        return JSONResponse({"error": f"original bytes missing in {fd}"}, status_code=400)
    data = original.read_bytes()
    from finetune_studio.data.prep import DataPrepRunner
    # Persist a DB row so this reprocess shows up in the activity feed.
    from finetune_studio import db
    db_row = db.create_data_prep_run(
        project_id=pid, filename=original.name, byte_count=len(data),
        source_id=source_id,
        settings_obj={"source": "reprocess", "source_id": source_id},
    )
    run_id = db_row["id"]
    progress_log: list[dict] = []
    def _cb(p):
        progress_log.append({
            "stage": p.stage, "pct": p.pct, "message": p.message,
            "source_id": p.source_id, "sha256": p.sha256,
            "chunks_total": p.chunks_total, "chunks_done": p.chunks_done,
            "qa_total": p.qa_total, "ts": time.time(),
        })
    runner = DataPrepRunner(pid=pid, data=data, filename=original.name,
                              uploaded_by="reprocess", progress_cb=_cb)
    _RUNS[(pid, run_id)] = {"runner": runner, "log": progress_log,
                              "filename": original.name, "byte_count": len(data)}
    async def _bg():
        try:
            db.mark_data_prep_running(run_id)
        except Exception:  # noqa: BLE001
            pass
        try:
            result = runner.run()
            progress_log.append({"stage": "done" if result.get("ok") else "error",
                                 "pct": 100 if result.get("ok") else 0,
                                 "message": json.dumps(result), "ts": time.time()})
            qa_total = 0
            for entry in progress_log:
                if isinstance(entry.get("qa_total"), int):
                    qa_total = max(qa_total, entry["qa_total"])
            if result.get("ok"):
                try:
                    db.mark_data_prep_done(
                        run_id, qa_total=qa_total, qa_approved=0,
                        output_path=result.get("output_path", "") or "",
                    )
                except Exception:  # noqa: BLE001
                    pass
            else:
                try:
                    db.mark_data_prep_failed(
                        run_id,
                        str(result.get("error") or result.get("message") or "reprocess failed"),
                    )
                except Exception:  # noqa: BLE001
                    pass
        except Exception as e:  # noqa: BLE001
            log.exception("data-prep reprocess failed")
            progress_log.append({"stage": "error", "pct": 0,
                                 "message": str(e), "ts": time.time()})
            try:
                db.mark_data_prep_failed(run_id, str(e))
            except Exception:  # noqa: BLE001
                pass
    import asyncio
    asyncio.get_event_loop().create_task(_bg())
    return {"ok": True, "run_id": run_id, "filename": original.name}
