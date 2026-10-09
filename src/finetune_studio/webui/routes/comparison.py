"""Compare tab — several models, the same questions, run then judge.

1. **Run** (``POST /api/compare/projects/{pid}/runs``) asks every chosen model every question of a quiz and saves the
   raw answers next to the answer key. It never scores: there is no verdict until step 2.
2. **Judge** (``POST .../groups/{gid}/judge``) lets one AI judge (any provider row, as on the Testing page) read all
   saved answers; a person can set any verdict with the Testing page's call
   ``PUT /api/testing/projects/{pid}/runs/{bid}/cases/{cid}/verdict``. ``GET .../groups/{gid}`` returns the side-by-side view.

Each model's answers are an ordinary test run (kind ``compare``), so the Testing page lists, judges and exports them too.
Stopping any of them (``POST /api/testing/projects/{pid}/runs/{bid}/cancel``) stops the whole group.
"""
from __future__ import annotations

import asyncio
from typing import Any

from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import HTMLResponse, JSONResponse

from finetune_studio import db
from finetune_studio.compare import session
from finetune_studio.testing.judge import JudgeUnavailable
from finetune_studio.testing.suite import load_test_suite
from finetune_studio.webui import compare_jobs, testing_jobs
from finetune_studio.webui.testing_models import BASE_MODEL_CHOICE, target_model

router = APIRouter()
pages = APIRouter()

MAX_MODELS = 6


async def _json_object(request: Request) -> dict:
    try:
        body = await request.json()
    except ValueError:
        raise HTTPException(status_code=400, detail="invalid JSON body") from None
    if not isinstance(body, dict):
        raise HTTPException(status_code=400, detail="body must be a JSON object")
    return body


def _project_or_404(pid: str) -> None:
    if not db.get_project(pid):
        raise HTTPException(status_code=404, detail="project not found")


def _busy(exc: testing_jobs.TestingBusy) -> JSONResponse:
    return JSONResponse({"error": str(exc), "active": exc.active}, status_code=409)


def _resolve_models(pid: str, raw: Any) -> list[compare_jobs.CompareModel]:
    """``models`` is a list of model entries (path, ``__base__`` or ``{path, label}``); each is resolved like the Testing page's."""
    from finetune_studio.naming import display_for_path

    if not isinstance(raw, list) or not 2 <= len(raw) <= MAX_MODELS:
        raise HTTPException(status_code=400, detail=f"models must list 2 to {MAX_MODELS} models to compare")
    out: list[compare_jobs.CompareModel] = []
    for entry in raw:
        choice, label = (str(entry.get("path") or ""), str(entry.get("label") or "")) if isinstance(entry, dict) \
            else (str(entry or ""), "")
        choice = choice.strip()
        if not choice:
            raise HTTPException(status_code=400, detail="every model needs a path (or __base__ for the untrained base)")
        try:
            path = target_model(pid, choice)
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from None
        if any(m.model_path == path for m in out):
            raise HTTPException(status_code=400, detail=f"the same model is listed twice: {path}")
        default = "Untrained base · " + display_for_path(path) if choice == BASE_MODEL_CHOICE else display_for_path(path)
        out.append(compare_jobs.CompareModel(label=label.strip() or default, model_path=path))
    return out


@router.post("/projects/{pid}/runs")
async def start_compare(pid: str, request: Request):
    """Start a comparison: 202 + the per-model run rows (``group_id`` is on each row's ``config.compare``)."""
    _project_or_404(pid)
    body = await _json_object(request)
    suite_path = str(body.get("suite_path") or "").strip()
    if not suite_path:
        return JSONResponse({"error": "suite_path required"}, status_code=400)
    try:
        cases = await asyncio.to_thread(load_test_suite, suite_path)
    except FileNotFoundError as exc:
        return JSONResponse({"error": str(exc)}, status_code=404)
    except (TypeError, ValueError, OSError) as exc:
        return JSONResponse({"error": f"invalid suite: {exc}"}, status_code=400)
    if not cases:
        return JSONResponse({"error": "suite has no usable cases"}, status_code=400)
    try:
        max_tokens, temperature = int(body.get("max_tokens", 512)), float(body.get("temperature", 0.3))
    except (TypeError, ValueError):
        raise HTTPException(status_code=400, detail="max_tokens and temperature must be numbers") from None
    auto = body.get("auto_judge")
    if auto is not None and not isinstance(auto, bool):
        raise HTTPException(status_code=400, detail="auto_judge must be true or false")
    models = _resolve_models(pid, body.get("models"))
    spec = compare_jobs.CompareSpec(
        project_id=pid, suite_name=suite_path.rsplit("/", 1)[-1], cases=cases, models=models, max_tokens=max_tokens,
        temperature=temperature, config={"suite_path": suite_path}, auto_judge=auto,
        judge_provider_id=str(body.get("judge_provider_id") or "").strip(),
    )
    try:
        rows = await compare_jobs.start_compare_job(spec)
    except testing_jobs.TestingBusy as exc:
        return _busy(exc)
    group_id = rows[0]["config"]["compare"]["group_id"]
    return JSONResponse({"ok": True, "group_id": group_id, "benchmarks": rows}, status_code=202)


@router.get("/projects/{pid}/groups")
async def list_groups(pid: str):
    """Saved comparisons of the project, newest first."""
    _project_or_404(pid)
    return await asyncio.to_thread(session.list_groups, pid)


@router.get("/projects/{pid}/groups/{gid}")
async def get_group(pid: str, gid: str):
    """Side-by-side view: per question every model's saved answer, the answer key, all verdicts (empty = awaiting)."""
    _project_or_404(pid)
    view = await asyncio.to_thread(session.side_by_side, pid, gid)
    if view is None:
        raise HTTPException(status_code=404, detail="comparison not found")
    return view


@router.post("/projects/{pid}/groups/{gid}/judge")
async def judge_group(pid: str, gid: str, request: Request):
    """Judge all saved answers of the group with one judge. ``{provider_id ('' = default), only_unjudged (default true)}``."""
    _project_or_404(pid)
    body = await _json_object(request)
    only_unjudged = body.get("only_unjudged", True)
    if not isinstance(only_unjudged, bool):
        raise HTTPException(status_code=400, detail="only_unjudged must be true or false")
    try:
        rows = await compare_jobs.start_group_judge(pid, gid, str(body.get("provider_id") or "").strip(),
                                                   only_unjudged=only_unjudged)
    except LookupError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from None
    except (ValueError, JudgeUnavailable) as exc:
        return JSONResponse({"error": str(exc)}, status_code=400)
    except testing_jobs.TestingBusy as exc:
        return _busy(exc)
    return JSONResponse({"ok": True, "group_id": gid, "benchmarks": rows}, status_code=202)


@router.post("/projects/{pid}/groups/{gid}/cancel")
async def cancel_group(pid: str, gid: str):
    """Stop the group's run or judge job at the next case; answers and verdicts saved so far are kept."""
    _project_or_404(pid)
    rows = await asyncio.to_thread(session.group_rows, pid, gid)
    if not rows:
        raise HTTPException(status_code=404, detail="comparison not found")
    if not any(testing_jobs.cancel(r["id"]) for r in rows):
        return JSONResponse({"error": "nothing is running for this comparison"}, status_code=409)
    return {"ok": True}


def library_models_for_compare(exclude: set[str], base_model: str) -> list[dict[str, Any]]:
    """Installed chat-capable models outside this project's exports: any trained export can be compared against any other
    installed model. The project's own base and exports are listed separately, so they are left out here."""
    from finetune_studio.models.registry import models_for_selectors
    from finetune_studio.webui.app import discovered_models

    out: list[dict[str, Any]] = []
    for m in models_for_selectors(discovered_models):
        path = str(getattr(m, "path", "") or "").rstrip("/\\")
        if not path or path in exclude or path == base_model.rstrip("/\\"):
            continue
        out.append({"name": m.name, "path": path, "format": m.format, "size_gb": m.size_gb})
    return out


# ── HTML page ────────────────────────────────────────────────────────────────


@pages.get("/projects/{pid}/compare", response_class=HTMLResponse)
async def compare_page(request: Request, pid: str):
    """Compare tab: pick models + a quiz, run, judge, read the answers side by side."""
    from finetune_studio.testing.judge import (
        default_judge_provider_id,
        list_judge_providers,
    )
    from finetune_studio.webui.routes.benchmarks import _discover_suites
    from finetune_studio.webui.routes.pages import _project_ctx, templates
    from finetune_studio.webui.testing_models import (
        default_model_path_for_testing,
        models_for_testing_page,
    )

    ctx = _project_ctx(pid)
    if not ctx:
        raise HTTPException(status_code=404, detail="Project not found")
    models = models_for_testing_page(ctx["project"].get("models") or [])
    return templates.TemplateResponse(request, "project_compare.html", {
        **ctx, "models": models, "default_model_path": default_model_path_for_testing(models),
        "library_models": library_models_for_compare({m["path"] for m in models}, str(ctx["project"].get("base_model") or "")),
        "suites": [s for s in _discover_suites(pid) if s.get("scoring") == "judge"],
        "judge_providers": list_judge_providers(), "default_judge_provider_id": default_judge_provider_id(),
        "max_models": MAX_MODELS,
    })
