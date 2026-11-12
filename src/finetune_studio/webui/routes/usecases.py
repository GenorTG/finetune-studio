"""Use-case templates API: ``/api/usecases/templates`` and ``/api/projects/{pid}/usecases*``.

Thin HTTP layer over ``finetune_studio.usecases``: build a use case from examples (SFT dataset + held-out suite),
start its base-vs-run evaluation, run the style A/B judge, read the paired report. Model calls (a distillation teacher)
and file writes run in a worker thread (``asyncio.to_thread``), never on the event loop.
"""
from __future__ import annotations

import asyncio
import json
import threading
from typing import Any

from fastapi import APIRouter, File, Form, UploadFile
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

from finetune_studio import db
from finetune_studio.usecases import evaluate, rows, store, teacher
from finetune_studio.usecases.schema import SchemaError
from finetune_studio.usecases.templates import TEMPLATES, get_template

router = APIRouter()

_TEACHER: dict[str, dict[str, Any]] = {}
_TEACHER_LOCK = threading.Lock()


class CreateBody(BaseModel):
    template: str
    name: str = ""
    rows: list[Any] = Field(default_factory=list)
    instruction: str = ""
    json_schema: dict[str, Any] | None = Field(default=None, alias="schema")
    labels: list[str] = Field(default_factory=list)
    style_guide: str = ""
    teacher: str = ""
    teacher_provider_id: str = ""
    seed: int = 42

    model_config = {"populate_by_name": True}


class EvaluateBody(BaseModel):
    run_id: str
    judge_provider_id: str = ""


class ABBody(BaseModel):
    judge_provider_id: str = ""


def _err(detail: str, status: int, **extra: Any) -> JSONResponse:
    return JSONResponse({"detail": detail, **extra}, status_code=status)


def _missing_project(pid: str) -> JSONResponse | None:
    return None if db.get_project(pid) else _err("project not found", 404)


@router.get("/usecases/templates")
async def list_templates() -> list[dict[str, Any]]:
    """The four templates: what each is for, the row shape, data needs, metric."""
    return [t.as_dict() for t in TEMPLATES.values()]


async def _create(pid: str, body: CreateBody) -> JSONResponse:
    try:
        template = get_template(body.template)
    except KeyError as exc:
        return _err(str(exc.args[0]), 400)
    if not body.rows:
        return _err("rows required: a list of examples (see GET /api/usecases/templates for the shape)", 400)
    examples = body.rows
    teacher_report = None
    if template.id == "distillation" and body.teacher_provider_id.strip():
        with _TEACHER_LOCK:
            if _TEACHER.get(pid, {}).get("running"):
                return _err("a teacher is already answering for this project", 409)
            _TEACHER[pid] = {"running": True, "done": 0, "total": 0}

        def on_progress(done: int, total: int) -> None:
            with _TEACHER_LOCK:
                _TEACHER[pid].update(done=done, total=total)

        try:
            chat, model = await asyncio.to_thread(teacher.open_teacher, body.teacher_provider_id.strip())
            examples, report = await asyncio.to_thread(teacher.fill_answers, examples, chat, body.instruction.strip(),
                                                       progress=on_progress)
        except teacher.TeacherUnavailable as exc:
            return _err(str(exc), 400)
        finally:
            with _TEACHER_LOCK:
                _TEACHER[pid]["running"] = False
        report.model = model
        teacher_report = report.as_dict()
        if report.asked and not report.answered:
            return _err("the teacher did not answer a single prompt", 502, teacher=teacher_report)
    opts = rows.BuildOptions(instruction=body.instruction, schema=body.json_schema, labels=body.labels,
                             style_guide=body.style_guide,
                             teacher=body.teacher or (teacher_report or {}).get("model", ""), seed=body.seed)
    try:
        prepared = await asyncio.to_thread(rows.prepare, template.id, examples, opts)
    except SchemaError as exc:
        return _err(f"schema: {exc}", 400)
    except rows.TooFewRows as exc:
        return _err(str(exc), 422, code="too_few_examples", errors=exc.errors, teacher=teacher_report)
    except rows.RowsError as exc:
        return _err(str(exc), 422, code="invalid_examples", errors=exc.errors, teacher=teacher_report)
    manifest = await asyncio.to_thread(store.create, pid, body.name, prepared, teacher_report=teacher_report)
    # ?usecase= fills the whole Training form (dataset + the plan's settings), not only the dataset picker.
    return JSONResponse({**manifest, "train_url": f"/projects/{pid}/training?usecase={manifest['id']}"},
                        status_code=201)


@router.post("/projects/{pid}/usecases", response_model=None)
async def create_usecase(pid: str, body: CreateBody) -> JSONResponse:
    """Build a use case from ``rows`` (JSON). 201 + manifest; 422 lists every unusable example."""
    return _missing_project(pid) or await _create(pid, body)


@router.post("/projects/{pid}/usecases/upload", response_model=None)
async def upload_usecase(pid: str, file: UploadFile = File(...), options: str = Form("{}")) -> JSONResponse:  # noqa: B008
    """Same as ``POST …/usecases`` with the examples in a file (JSON list, ``{"rows": [...]}`` or JSONL) and the
    other fields as a JSON object in the ``options`` form field."""
    missing = _missing_project(pid)
    if missing is not None:
        return missing
    try:
        opts = json.loads(options or "{}")
        if not isinstance(opts, dict):
            return _err("invalid options: must be a JSON object", 400)
        examples = rows.parse_upload(await file.read())
        body = CreateBody(**{**opts, "rows": examples})
    except rows.RowsError as exc:
        return _err(str(exc), 400)
    except ValueError as exc:  # json / pydantic validation
        return _err(f"invalid options: {exc}", 400)
    return await _create(pid, body)


@router.get("/projects/{pid}/usecases/teacher/progress", response_model=None)
async def teacher_progress(pid: str) -> dict[str, Any] | JSONResponse:
    """Live progress of a teacher writing distillation answers (``running: false`` when none)."""
    missing = _missing_project(pid)
    if missing is not None:
        return missing
    with _TEACHER_LOCK:
        return dict(_TEACHER.get(pid) or {"running": False})


@router.get("/projects/{pid}/usecases", response_model=None)
async def list_usecases(pid: str) -> list[dict[str, Any]] | JSONResponse:
    missing = _missing_project(pid)
    if missing is not None:
        return missing
    return await asyncio.to_thread(store.list_all, pid)


@router.get("/projects/{pid}/usecases/{uid}", response_model=None)
async def get_usecase(pid: str, uid: str) -> dict[str, Any] | JSONResponse:
    missing = _missing_project(pid)
    if missing is not None:
        return missing
    try:
        return await asyncio.to_thread(store.get, pid, uid)
    except LookupError as exc:
        return _err(str(exc), 404)


@router.delete("/projects/{pid}/usecases/{uid}", response_model=None)
async def delete_usecase(pid: str, uid: str) -> dict[str, Any] | JSONResponse:
    """Remove the manifest, the held-out suite and A/B results. The dataset and any trained runs stay."""
    missing = _missing_project(pid)
    if missing is not None:
        return missing
    try:
        return await asyncio.to_thread(store.delete, pid, uid)
    except LookupError as exc:
        return _err(str(exc), 404)


@router.post("/projects/{pid}/usecases/{uid}/evaluate", response_model=None)
async def start_evaluation(pid: str, uid: str, body: EvaluateBody) -> JSONResponse:
    """Untrained base and the run answer the held-out questions (one compare group). 202 + the evaluation."""
    from finetune_studio.webui.testing_jobs import TestingBusy

    missing = _missing_project(pid)
    if missing is not None:
        return missing
    try:
        record = await asyncio.to_thread(evaluate.start, pid, uid, body.run_id.strip(),
                                         judge_provider_id=body.judge_provider_id.strip())
    except LookupError as exc:
        return _err(str(exc), 404)
    except evaluate.EvaluationConflict as exc:
        return _err(str(exc), 409)
    except TestingBusy as exc:
        return _err(str(exc), 409, code="busy", active=exc.active)
    return JSONResponse({"ok": True, **record}, status_code=202)


@router.get("/projects/{pid}/usecases/{uid}/evaluations/{gid}", response_model=None)
async def evaluation_report(pid: str, uid: str, gid: str) -> dict[str, Any] | JSONResponse:
    """Paired base-vs-run result: per-model metric, McNemar (or the A/B sign test), notes."""
    missing = _missing_project(pid)
    if missing is not None:
        return missing
    try:
        return await asyncio.to_thread(evaluate.report, pid, uid, gid)
    except LookupError as exc:
        return _err(str(exc), 404)


@router.post("/projects/{pid}/usecases/{uid}/evaluations/{gid}/ab", response_model=None)
async def start_ab(pid: str, uid: str, gid: str, body: ABBody) -> JSONResponse:
    """Style template: start the blind A/B judge on a finished evaluation. 202 + its state."""
    from finetune_studio.testing.judge import JudgeUnavailable
    from finetune_studio.webui.testing_jobs import TestingBusy

    missing = _missing_project(pid)
    if missing is not None:
        return missing
    try:
        state = await asyncio.to_thread(evaluate.start_ab, pid, uid, gid, body.judge_provider_id.strip())
    except LookupError as exc:
        return _err(str(exc), 404)
    except evaluate.EvaluationConflict as exc:
        return _err(str(exc), 409)
    except TestingBusy as exc:
        return _err(str(exc), 409, code="busy", active=exc.active)
    except (ValueError, JudgeUnavailable) as exc:
        return _err(str(exc), 400)
    return JSONResponse({"ok": True, **state}, status_code=202)
