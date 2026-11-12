"""Use-case drafting API: ``/api/projects/{pid}/usecases/drafts*`` (``usecases/drafting.py``).

Upload raw texts, the AI assistant's ``drafting`` job drafts the missing half of each example, the user approves or edits
every row, and ``…/build`` turns the approved rows into a use case through the normal build. Assistant calls run in a
worker thread (``asyncio.to_thread``); one draft run per project; ``GET …/drafts/progress`` feeds the status line.
Included before ``usecases.router`` so ``drafts`` is never read as a use-case id.
"""
from __future__ import annotations

import asyncio
import json
import threading
from typing import Any

from fastapi import APIRouter, File, Form, UploadFile
from fastapi.responses import JSONResponse
from pydantic import BaseModel

from finetune_studio import db
from finetune_studio.data.prep.preference import NoGenerator
from finetune_studio.usecases import drafting
from finetune_studio.usecases.schema import SchemaError

router = APIRouter()

_JOBS: dict[str, dict[str, Any]] = {}
_JOBS_LOCK = threading.Lock()


class RowChange(BaseModel):
    id: str
    status: str | None = None
    draft: Any = None
    set_draft: bool = False   # true when ``draft`` carries an edit


class RowsBody(BaseModel):
    changes: list[RowChange]


class BuildBody(BaseModel):
    name: str = ""
    instruction: str = ""
    seed: int = 42


def _err(detail: str, status: int, **extra: Any) -> JSONResponse:
    return JSONResponse({"detail": detail, **extra}, status_code=status)


def _missing(pid: str) -> JSONResponse | None:
    return None if db.get_project(pid) else _err("project not found", 404)


def _draft_options(raw: dict[str, Any]) -> drafting.DraftOptions:
    labels = raw.get("labels") or []
    fields = raw.get("fields") or []
    if isinstance(labels, str):
        labels = [x.strip() for x in labels.split(",")]
    if isinstance(fields, str):
        fields = [x.strip() for x in fields.split(",")]
    schema = raw.get("schema")
    if schema is not None and not isinstance(schema, dict):
        raise drafting.DraftError("schema must be a JSON object")
    return drafting.DraftOptions(labels=[str(x) for x in labels if str(x).strip()],
                                 fields=[str(x) for x in fields if str(x).strip()], schema=schema,
                                 style_guide=str(raw.get("style_guide") or ""), hint=str(raw.get("hint") or ""))


def _run_drafts(pid: str, template: str, texts: list[str], opts: drafting.DraftOptions, name: str,
                progress: Any) -> dict[str, Any]:
    from finetune_studio.data.mining.scheduler import helper_info
    from finetune_studio.data.prep.generator import helper_resolution_error
    from finetune_studio.data.prep.preference import helper_generate

    generate = helper_generate(max_tokens=600 if template == "extraction" else 200, temperature=0.2, job="drafting")
    if generate is None:
        raise NoGenerator(helper_resolution_error("drafting"))
    info = helper_info("drafting")
    rows = drafting.draft_rows(template, texts, generate, opts, workers=int(info.get("concurrency") or 1),
                               progress=progress)
    assistant = {k: info.get(k) for k in ("provider_id", "label", "model", "instructions_tag")}
    return drafting.create(pid, name, template, opts, rows, assistant)


@router.post("/projects/{pid}/usecases/drafts", response_model=None)
async def create_drafts(pid: str, file: UploadFile = File(...), options: str = Form("{}")) -> JSONResponse:  # noqa: B008
    """Draft examples for the texts in ``file`` (JSON list, JSONL or plain text). ``options``: a JSON object with
    ``template`` (classification | extraction | style), ``labels`` / ``fields`` / ``schema`` / ``style_guide``, ``hint``
    (what the texts are) and ``name``. 201 + the draft set; 409 when the assistant is not usable now."""
    missing = _missing(pid)
    if missing is not None:
        return missing
    try:
        raw = json.loads(options or "{}")
        if not isinstance(raw, dict):
            return _err("options must be a JSON object", 400)
        template = str(raw.get("template") or "")
        opts = _draft_options(raw)
        drafting.check_options(template, opts)
        texts = drafting.parse_texts(await file.read(), file.filename or "")
    except (drafting.DraftError, SchemaError) as exc:
        return _err(str(exc), 400)
    except ValueError as exc:
        return _err(f"invalid options: {exc}", 400)
    with _JOBS_LOCK:
        if _JOBS.get(pid, {}).get("running"):
            return _err("the assistant is already drafting for this project", 409)
        _JOBS[pid] = {"running": True, "done": 0, "planned": len(texts), "drafted": 0}

    def on_progress(update: dict[str, Any]) -> None:
        with _JOBS_LOCK:
            _JOBS[pid].update(update)

    try:
        data = await asyncio.to_thread(_run_drafts, pid, template, texts, opts, str(raw.get("name") or ""),
                                       on_progress)
    except NoGenerator as exc:
        return _err(str(exc), 409, code="no_model")
    except drafting.DraftError as exc:
        return _err(str(exc), 502, code="assistant_failed")
    finally:
        with _JOBS_LOCK:
            _JOBS[pid]["running"] = False
    return JSONResponse(data, status_code=201)


@router.get("/projects/{pid}/usecases/drafts/progress", response_model=None)
async def drafts_progress(pid: str) -> dict[str, Any] | JSONResponse:
    missing = _missing(pid)
    if missing is not None:
        return missing
    with _JOBS_LOCK:
        return dict(_JOBS.get(pid) or {"running": False})


@router.get("/projects/{pid}/usecases/drafts", response_model=None)
async def list_drafts(pid: str) -> dict[str, Any] | JSONResponse:
    return _missing(pid) or {"drafts": await asyncio.to_thread(drafting.list_all, pid)}


@router.get("/projects/{pid}/usecases/drafts/{did}", response_model=None)
async def get_drafts(pid: str, did: str) -> dict[str, Any] | JSONResponse:
    missing = _missing(pid)
    if missing is not None:
        return missing
    try:
        data = await asyncio.to_thread(drafting.get, pid, did)
    except LookupError as exc:
        return _err(str(exc), 404)
    return {**data, "counts": drafting.counts(data["rows"])}


@router.patch("/projects/{pid}/usecases/drafts/{did}/rows", response_model=None)
async def update_draft_rows(pid: str, did: str, body: RowsBody) -> dict[str, Any] | JSONResponse:
    """Approve / reject / edit rows: ``{changes: [{id, status?, draft?, set_draft?}]}``. All or nothing."""
    missing = _missing(pid)
    if missing is not None:
        return missing
    changes = []
    for c in body.changes:
        change: dict[str, Any] = {"id": c.id}
        if c.status is not None:
            change["status"] = c.status
        if c.set_draft:
            change["draft"] = c.draft
        changes.append(change)
    try:
        data = await asyncio.to_thread(drafting.update_rows, pid, did, changes)
    except LookupError as exc:
        return _err(str(exc), 404)
    except drafting.DraftError as exc:
        return _err(str(exc), 400)
    return {**data, "counts": drafting.counts(data["rows"])}


@router.post("/projects/{pid}/usecases/drafts/{did}/build", response_model=None)
async def build_from_drafts(pid: str, did: str, body: BuildBody) -> JSONResponse:
    """Build a use case from the APPROVED rows only, through the normal build (same checks, held-out split, plan)."""
    from finetune_studio.webui.routes.usecases import CreateBody, _create

    missing = _missing(pid)
    if missing is not None:
        return missing
    try:
        data = await asyncio.to_thread(drafting.get, pid, did)
    except LookupError as exc:
        return _err(str(exc), 404)
    examples = drafting.approved_examples(data)
    if not examples:
        return _err("approve some drafts first: only approved rows become examples", 422, code="nothing_approved")
    opts = drafting.options_of(data)
    create = CreateBody(template=data["template"], name=body.name or data.get("name") or "", rows=examples,
                        instruction=body.instruction, labels=opts.labels if data["template"] == "classification" else [],
                        style_guide=opts.style_guide, seed=body.seed,
                        **({"schema": opts.schema} if data["template"] == "extraction" and opts.schema else {}))
    return await _create(pid, create)


@router.delete("/projects/{pid}/usecases/drafts/{did}", response_model=None)
async def delete_drafts(pid: str, did: str) -> dict[str, Any] | JSONResponse:
    missing = _missing(pid)
    if missing is not None:
        return missing
    try:
        await asyncio.to_thread(drafting.delete, pid, did)
    except LookupError as exc:
        return _err(str(exc), 404)
    return {"ok": True}
