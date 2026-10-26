"""Routes for the AI-powered Data Prep workflow + Model provider management.

Files enter through the file library and are parsed once (``knowledge.sources``); this module lists those sources
for the pairs page and owns the Q&A pair review/export (``qa/pairs/<id>.json``).
"""

from __future__ import annotations

import asyncio
import json
import logging
import time
from pathlib import Path

from fastapi import APIRouter, Request
from fastapi.responses import HTMLResponse, JSONResponse, Response

from finetune_studio import __release_channel__ as RELEASE_CHANNEL
from finetune_studio import __version__ as APP_VERSION

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
        return JSONResponse({"detail": "project not found"}, status_code=404)
    return None

_QA_EDITABLE_FIELDS = frozenset({"question", "answer", "status", "chunk_idx", "note"})


# ── HTML page ────────────────────────────────────────────────────────────

@_pages.get("/projects/{pid}/data-prep", response_class=HTMLResponse)
async def data_prep_page(request: Request, pid: str):
    from fastapi.templating import Jinja2Templates

    from finetune_studio import db
    from finetune_studio.webui.app import discovered_models
    templates = Jinja2Templates(directory=str(Path(__file__).parent.parent / "templates"))
    templates.env.globals["app_version"] = APP_VERSION
    templates.env.globals["release_channel"] = RELEASE_CHANNEL
    from finetune_studio.webui.flow_next import flow_next_step
    templates.env.globals["flow_next_step"] = flow_next_step
    project = db.get_project(pid)
    if not project:
        return HTMLResponse("Project not found", status_code=404)
    from finetune_studio.data.parsers import readable_extensions
    from finetune_studio.models.helper import (
        DEFAULT_HELPER_LABEL,
        get_configured_helper_provider,
        get_helper_provider_id,
    )
    helper = get_configured_helper_provider("mining")
    ctx = {
        "request": request,
        "pid": pid,
        "project": project,
        "models": discovered_models,
        "helper_label": (helper or {}).get("label") or DEFAULT_HELPER_LABEL,
        "helper_provider_id": get_helper_provider_id("mining"),
        # Every parser-supported upload is parsed on arrival; "Use as source" re-queues one that is not.
        "promotable_extensions": sorted(readable_extensions()),
    }
    return templates.TemplateResponse(request, "data_prep.html", ctx)


# RAG page lives in pages.project_brain_page (passes indexed_docs).


# ── Provider CRUD ───────────────────────────────────────────────────────

@router.get("/providers")
async def list_providers():
    from finetune_studio.models.assistant import jobs_state
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
        # the main assistant (kept under its helper-era names) and the connection each job runs on
        "helper": helper,
        "helper_provider_id": get_helper_provider_id(),
        "helper_label": (helper or {}).get("label") or DEFAULT_HELPER_LABEL,
        "jobs": {j["id"]: {"provider_id": j["provider_id"], "kind": j["provider_kind"], "label": j["provider_label"]}
                 for j in jobs_state()},
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
             "detail": helper_missing_message(missing)},
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
            return JSONResponse({"ok": False, "detail": str(e)}, status_code=400)


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
        return JSONResponse({"ok": False, "detail": f"Not a GGUF file: {path or '(empty)'}"}, status_code=400)
    get_manager().upsert_provider(id=DEFAULT_HELPER_PROVIDER_ID, model_id=str(resolved))
    return {"ok": True, "path": str(resolved)}


@router.post("/providers/unload")
async def unload_active():
    """Unload the active provider (both engines — see E2E-22)."""
    from finetune_studio.models.llama_loader import unload_all_models
    from finetune_studio.smart_brain.model_cache import release_brain_models
    await asyncio.to_thread(unload_all_models)  # seconds for a resident 12B: never on the event loop
    await asyncio.to_thread(release_brain_models, "unload all models")
    return {"ok": True, "active": None}


# ── Data Prep: sources, curate, export ──────────────────────────────────

@router.get("/projects/{pid}/data-prep/sources")
async def list_sources_route(pid: str):
    """Every parsed (or parsing) file of the project with its mining coverage."""
    missing = _project_404(pid)
    if missing is not None:
        return missing
    from finetune_studio.data.mining import project_coverage
    from finetune_studio.knowledge.sources import list_sources

    def _build() -> list[dict]:
        cov = {c.source_id: c.as_dict() for c in project_coverage(pid)}
        return [{**s.as_dict(), "parse_status": s.status, "mining": cov.get(s.id)} for s in list_sources(pid)]
    return {"sources": await asyncio.to_thread(_build)}


@router.post("/projects/{pid}/data-prep/sources")
async def parse_source_route(pid: str, request: Request):
    """Parse one library file now (``{file_id}``): an upload whose parse failed or that predates auto-parsing.
    Idempotent: a ready source is returned as it is."""
    missing = _project_404(pid)
    if missing is not None:
        return missing
    from finetune_studio.knowledge import sources

    body = await request.json()
    fid = str(body.get("file_id") or "")
    if not fid:
        return JSONResponse({"detail": "file_id required"}, status_code=400)
    source = sources.stage(pid, fid)
    source = await asyncio.to_thread(sources.parse, pid, source.id)
    if source.status == sources.ERROR:
        return JSONResponse({"detail": source.error, "source": source.as_dict()}, status_code=422)
    return {"ok": True, "source": source.as_dict()}


@router.get("/projects/{pid}/data-prep/qa")
async def list_qa_route(pid: str, source_id: str | None = None, status: str | None = None):
    missing = _project_404(pid)
    if missing is not None:
        return missing
    from finetune_studio.data.fs import qa as qa_fs
    return {"items": await asyncio.to_thread(qa_fs.list_qa_pairs, pid, source_id=source_id, status=status)}


@router.get("/projects/{pid}/data-prep/qa/queue")
async def qa_queue_route(pid: str, status: str | None = None, source_id: str | None = None, q: str = "",
                         offset: int = 0, limit: int = 200):
    """One page of the review queue: slim rows in document order, project-wide counts, per-file counts (no chunk text)."""
    missing = _project_404(pid)
    if missing is not None:
        return missing
    from finetune_studio.data.fs.qa_review import review_queue
    return await asyncio.to_thread(review_queue, pid, status=status or None, source_id=source_id or None, text=q,
                                   offset=offset, limit=limit)


@router.get("/projects/{pid}/data-prep/qa/{qa_id}/context")
async def qa_context_route(pid: str, qa_id: str):
    """A pair with its full source chunk, for the review panel."""
    missing = _project_404(pid)
    if missing is not None:
        return missing
    from finetune_studio.data.fs.qa_review import pair_context
    ctx = await asyncio.to_thread(pair_context, pid, qa_id)
    if ctx is None:
        return JSONResponse({"detail": "not found"}, status_code=404)
    return ctx


@router.post("/projects/{pid}/data-prep/qa")
async def add_qa_route(pid: str, request: Request):
    """Add a pair the reviewer wrote (a fact the miner never stated). Body: source_id, chunk_idx, question, answer."""
    missing = _project_404(pid)
    if missing is not None:
        return missing
    body = await request.json()
    if not isinstance(body, dict):
        return JSONResponse({"detail": "JSON object body required"}, status_code=400)
    question, answer = str(body.get("question") or "").strip(), str(body.get("answer") or "").strip()
    if not question or not answer or not body.get("source_id"):
        return JSONResponse({"detail": "source_id, question and answer are required"}, status_code=400)
    from finetune_studio.data.fs.qa_review import add_reviewer_pair
    try:
        return await asyncio.to_thread(add_reviewer_pair, pid, str(body["source_id"]), int(body.get("chunk_idx") or 1),
                                       question, answer)
    except KeyError as e:
        return JSONResponse({"detail": str(e)}, status_code=404)
    except ValueError as e:
        return JSONResponse({"detail": str(e)}, status_code=400)


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
    from finetune_studio.data.fs import qa as qa_fs
    from finetune_studio.data.prep.dedupe import find_near_duplicates

    pairs = await asyncio.to_thread(qa_fs.list_qa_pairs, pid, source_id=body.get("source_id") or None, status="pending")
    pairs.sort(key=lambda p: (str(p.get("source_id")), int(p.get("chunk_idx") or 0), float(p.get("created_at") or 0), str(p.get("id"))))
    dupes = find_near_duplicates(pairs)
    if not body.get("dry_run"):
        def _apply() -> None:
            for dup_id, kept_id in dupes:
                qa_fs.update_qa_pair(pid, dup_id, status="rejected", note=f"near-duplicate of {kept_id}")
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
        return JSONResponse({"detail": "JSON object body required"}, status_code=400)
    fields = {k: v for k, v in body.items() if k in _QA_EDITABLE_FIELDS}
    if not fields:
        return JSONResponse(
            {"detail": f"no editable fields; allowed: {sorted(_QA_EDITABLE_FIELDS)}"},
            status_code=400,
        )
    from finetune_studio.data.fs import qa as qa_fs
    if "status" in fields:
        fields["reviewed_at"] = time.time()  # a human verdict (approve/reject/reset) — told apart from machine auto-approval
    updated = await asyncio.to_thread(qa_fs.update_qa_pair, pid, qa_id, **fields)
    if updated is None:
        return JSONResponse({"detail": "not found"}, status_code=404)
    return updated


@router.post("/projects/{pid}/data-prep/qa/bulk")
async def bulk_action(pid: str, request: Request):
    missing = _project_404(pid)
    if missing is not None:
        return missing
    body = await request.json()
    ids, action = body.get("ids", []), body.get("action", "")
    new_status = {"approve": "approved", "reject": "rejected", "reset": "pending"}.get(action)
    if not new_status:
        return JSONResponse({"detail": f"unknown action: {action}"}, status_code=400)
    note = body.get("note")
    from finetune_studio.data.fs import qa as qa_fs
    from finetune_studio.data.fs.qa_review import status_counts

    def _apply() -> tuple[int, dict[str, int]]:
        extra = {"note": str(note)} if note is not None else {}
        done = sum(1 for qa_id in ids
                   if qa_fs.update_qa_pair(pid, qa_id, status=new_status, reviewed_at=time.time(), **extra) is not None)
        return done, status_counts(qa_fs.list_qa_pairs(pid))

    updated, counts = await asyncio.to_thread(_apply)
    return {"ok": True, "updated": updated, "counts": counts}


@router.get("/projects/{pid}/data-prep/export")
async def export_qa(pid: str, fmt: str = "sharegpt", only: str = "approved",
                    force: bool = False, grounded_share: float | None = None, distractors: int = 0):
    """Export pairs as JSONL + register the dataset.

    Blocked (409, with the incomplete files) until every file is complete: each data point of its inventory stated by an
    approved pair or excluded (``data.prep.dataset_build.completeness_gate``); ``force=true`` exports anyway.
    ``grounded_share`` (0-1): fraction of rows rewritten to carry the RAG-chat prompt + CONTEXT from the pair's own source
    chunk. Omitted = auto (40% when the project has a built RAG corpus, else off); ``0`` = plain rows only.
    ``distractors`` (0-2): extra other-file chunks in that CONTEXT.
    """
    missing = _project_404(pid)
    if missing is not None:
        return missing
    if only not in ("approved", "pending", "rejected", "all"):
        return JSONResponse({"detail": f"unknown only filter: {only}"}, status_code=400)
    from finetune_studio.data.prep.dataset_build import (
        ExportBlocked,
        completeness_gate,
        persist_export,
    )
    from finetune_studio.data.prep.export import build_qa_export
    from finetune_studio.data.prep.grounding import resolve_grounding
    try:
        grounding = resolve_grounding(pid, grounded_share, distractors)
    except ValueError as e:
        return JSONResponse({"detail": str(e)}, status_code=400)
    try:
        await asyncio.to_thread(completeness_gate, pid, force=force)
    except ExportBlocked as blocked:
        return JSONResponse({"detail": str(blocked), "incomplete_files": blocked.files,
                             "hint": "pass ?force=true to export anyway"}, status_code=409)
    try:
        result = await asyncio.to_thread(build_qa_export, pid, fmt, only, grounding=grounding)
    except ValueError as e:
        return JSONResponse({"detail": str(e)}, status_code=400)
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


# ── Audit ───────────────────────────────────────────────────

@router.get("/projects/{pid}/data-prep/audit")
async def data_prep_audit(pid: str) -> dict:
    """Return deterministic raw-file and curated-dataset fidelity evidence."""
    missing = _project_404(pid)
    if missing is not None:
        return missing
    from finetune_studio.data.audit import audit_project_sources, audit_qa_pairs

    return {"sources": audit_project_sources(pid), "dataset": audit_qa_pairs(pid)}
