# mypy: disable-error-code="arg-type,call-arg"
# Reason: starlette's Jinja2Templates.TemplateResponse typing stubs use the
# newer signature `TemplateResponse(request, name, context)` but the common
# pattern (used here) is `TemplateResponse(name, context_dict)` where context
# contains "request". This is a long-standing stubs issue; see
# https://github.com/encode/starlette/issues/1426
"""Main page layouts (home, settings, help).

WHAT THIS FILE DOES
===================
Defines the HTML page routes for the finetune-studio web UI:
  - GET /          → home page (index.html)
  - GET /models    → model browser
  - GET /training  → training dashboard
  - GET /data      → data files
  - GET /testing   → testing/inference playground

KEY CONCEPTS
============
- FastAPI route handlers: async functions returning HTML responses.
- Jinja2Templates: starlette's templating engine for rendering Jinja2 templates.
- Per-route # type: ignore: starlette's TemplateResponse has a known typing quirk
  where the modern signature requires positional Request as first arg, but the
  common pattern (used here) is `TemplateResponse(name, context_dict)` where
  context_dict contains "request". This is a long-standing stubs issue, not a
  real bug. See: https://github.com/encode/starlette/issues/1426
"""

from pathlib import Path

from fastapi import APIRouter, Request
from fastapi.responses import HTMLResponse, RedirectResponse
from fastapi.templating import Jinja2Templates

templates = Jinja2Templates(directory=str(Path(__file__).parent.parent / "templates"))
router = APIRouter()


@router.get("/", response_class=HTMLResponse)
async def index(request: Request):
    """Home page with overview dashboard."""
    from finetune_studio.webui.app import discovered_models, training_engine
    return templates.TemplateResponse(
        request,
        "index.html",
        {
            "models": discovered_models,
            "training_state": training_engine.state,
        },
    )


@router.get("/models", response_class=HTMLResponse)
async def models_page(request: Request):
    """Model browser page."""
    from finetune_studio.webui.app import discovered_models
    return templates.TemplateResponse(
        request,
        "models.html",
        {"request": request, "models": discovered_models},
    )


@router.get("/training", response_class=HTMLResponse)
async def training_page(request: Request):
    """Training dashboard page."""
    from finetune_studio.webui.app import discovered_models, training_engine
    return templates.TemplateResponse(
        request,
        "training.html",
        {
            "request": request,
            "models": discovered_models,
            "training_state": training_engine.state,
        },
    )


@router.get("/data", response_class=HTMLResponse)
async def data_page(request: Request):
    """Data files browser page."""
    from finetune_studio.config import settings
    from finetune_studio.data.organizer import scan_data_files
    files = scan_data_files(settings.data_dir)
    return templates.TemplateResponse(
        request,
        "data.html",
        {"request": request, "files": files},
    )


@router.get("/testing", response_class=HTMLResponse)
async def testing_page(request: Request):
    """Testing/inference playground page."""
    from finetune_studio.webui.app import discovered_models, inference_engine
    return templates.TemplateResponse(
        request,
        "testing.html",
        {
            "request": request,
            "models": discovered_models,
            "inference_engine": inference_engine,
        },
    )


@router.get("/projects", response_class=HTMLResponse)
async def projects_page(request: Request):
    """Project list / create page."""
    from finetune_studio import db
    projects = db.list_projects()
    return templates.TemplateResponse(
        request,
        "projects.html",
        {"request": request, "projects": projects},
    )


@router.get("/projects/{pid}/data/{dataset_path:path}", response_class=HTMLResponse)
async def data_editor_page(request: Request, pid: str, dataset_path: str):
    """Project-scoped data editor for a JSONL dataset."""
    from finetune_studio import db as _db
    project = _db.get_project(pid)
    if not project:
        return RedirectResponse(url="/projects", status_code=302)
    return templates.TemplateResponse(
        request,
        "data_editor.html",
        {
            "request": request,
            "project": project,
            "dataset": dataset_path,
        },
    )


@router.get("/projects/{pid}/benchmarks", response_class=HTMLResponse)
async def benchmarks_page(request: Request, pid: str):
    """Benchmarks tab — run suites, view scores, compare runs."""
    from finetune_studio import db
    from finetune_studio.webui.routes.benchmarks import _discover_suites, _latest_benchmark
    project = db.get_project(pid)
    if not project:
        return RedirectResponse(url="/projects", status_code=302)
    runs = db.list_runs(pid)
    suites = _discover_suites()
    # Augment runs with latest benchmark
    for run in runs:
        run["latest_benchmark"] = _latest_benchmark(run["id"])
    # Build flat list of all benchmarks across all runs for history table
    all_benchmarks = []
    run_name_map = {r["id"]: r["name"] for r in runs}
    import time as _time
    for run in runs:
        for b in db.list_benchmarks(run["id"]):
            b["_run_name"] = run_name_map.get(run["id"], run["id"])
            b["_ran_at_str"] = _time.strftime("%Y-%m-%d %H:%M", _time.localtime(b["ran_at"]))
            all_benchmarks.append(b)
    all_benchmarks.sort(key=lambda x: x["ran_at"], reverse=True)
    # Default comparison: first two runs
    comparison_runs = [runs[0]["id"], runs[1]["id"]] if len(runs) >= 2 else []
    return templates.TemplateResponse(
        request,
        "benchmarks.html",
        {
            "request": request,
            "project": project,
            "runs": runs,
            "suites": suites,
            "all_benchmarks": all_benchmarks,
            "comparison_runs": comparison_runs,
        },
    )


@router.get("/projects/{pid}", response_class=HTMLResponse)
async def project_detail_page(request: Request, pid: str):
    """Project detail — RAGs + Runs."""
    from finetune_studio import db
    from finetune_studio.webui.app import discovered_models
    project = db.get_project(pid)
    if not project:
        return RedirectResponse(url="/projects", status_code=302)
    project["rags"] = db.list_rags(pid)
    project["runs"] = db.list_runs(pid)
    for run in project["runs"]:
        run["benchmarks"] = db.list_benchmarks(run["id"])
    return templates.TemplateResponse(
        request,
        "project.html",
        {
            "request": request,
            "project": project,
            "models": discovered_models,
        },
    )


@router.get("/chat", response_class=HTMLResponse)
async def chat_page(request: Request):
    """Inference chat page (user picks project in-page)."""
    return templates.TemplateResponse(request, "chat_v2.html", {"request": request})


@router.get("/projects/{pid}/chat", response_class=HTMLResponse)
async def project_chat_page(request: Request, pid: str):
    """Deep-link to chat with a specific project pre-selected."""
    from finetune_studio import db
    project = db.get_project(pid)
    if not project:
        return RedirectResponse(url="/chat", status_code=302)
    return templates.TemplateResponse(request, "chat_v2.html", {"request": request})


@router.get("/projects/{pid}/agentic", response_class=HTMLResponse)
async def agentic_page(request: Request, pid: str):
    """Agentic Tools playground page."""
    from finetune_studio import db
    from finetune_studio.webui.app import discovered_models
    project = db.get_project(pid)
    if not project:
        return RedirectResponse(url="/projects", status_code=302)
    project["rags"] = db.list_rags(pid)
    return templates.TemplateResponse(
        request,
        "agentic.html",
        {
            "request": request,
            "project": project,
            "models": discovered_models,
        },
    )
