# mypy: disable-error-code="arg-type,call-arg"
# Reason: starlette's Jinja2Templates.TemplateResponse typing stubs use the
# newer signature `TemplateResponse(request, name, context)` but the common
# pattern (used here) is `TemplateResponse(name, context_dict)` where context
# contains "request". This is a long-standing stubs issue; see
# https://github.com/encode/starlette/issues/1426
"""Page routes — dashboard + project-scoped pages.

DASHBOARD (/)
  System overview only: project list, system stats.

PROJECT-SCOPED PAGES (/projects/{pid}/...)
  Data, Training, Testing, Models — only accessible inside a project.
"""

from pathlib import Path

from fastapi import APIRouter, Request
from fastapi.responses import HTMLResponse, RedirectResponse
from fastapi.templating import Jinja2Templates

templates = Jinja2Templates(directory=str(Path(__file__).parent.parent / "templates"))
router = APIRouter()


# ── Helpers ──────────────────────────────────────────────────────────────

def _require_project(pid: str):
    """Return project dict or None (caller redirects to /projects)."""
    from finetune_studio import db
    return db.get_project(pid)


def _project_ctx(pid: str) -> dict:
    """Build common template context for project pages."""
    from finetune_studio import db
    project = db.get_project(pid)
    if not project:
        return {}
    return {"project": project, "pid": pid}


# ── Dashboard ────────────────────────────────────────────────────────────

@router.get("/", response_class=HTMLResponse)
async def index(request: Request):
    """Home page — project list + system overview."""
    from finetune_studio import db
    from finetune_studio.webui.app import discovered_models, training_engine
    projects = db.list_projects()
    return templates.TemplateResponse(
        request,
        "index.html",
        {
            "projects": projects,
            "models": discovered_models,
            "training_state": training_engine.state,
        },
    )


# ── Global Inference ──────────────────────────────────────────────────

@router.get("/inference", response_class=HTMLResponse)
async def inference_page(request: Request):
    """Global inference page — load any model, chat, run benchmarks."""
    from finetune_studio.webui.app import discovered_models
    return templates.TemplateResponse(
        request,
        "inference.html",
        {"request": request, "models": discovered_models},
    )


@router.get("/models/explore", response_class=HTMLResponse)
async def hf_models_page(request: Request):
    """HuggingFace model browser + downloader (LM Studio-style)."""
    from finetune_studio.webui.app import discovered_models
    return templates.TemplateResponse(
        request,
        "hf_models.html",
        {"request": request, "models": discovered_models},
    )


# ── Projects list ────────────────────────────────────────────────────────

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


# ── Project detail ───────────────────────────────────────────────────────

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
            "pid": pid,
            "project": project,
            "models": discovered_models,
        },
    )


# ── Project-scoped pages ────────────────────────────────────────────────

@router.get("/projects/{pid}/data", response_class=HTMLResponse)
async def project_data_page(request: Request, pid: str):
    """Data files for a project."""
    from finetune_studio.config import settings
    from finetune_studio.data.organizer import scan_data_files
    ctx = _project_ctx(pid)
    if not ctx:
        return RedirectResponse(url="/projects", status_code=302)
    files = scan_data_files(settings.data_dir)
    return templates.TemplateResponse(
        request,
        "project_data.html",
        {**ctx, "files": files},
    )


@router.get("/projects/{pid}/training", response_class=HTMLResponse)
async def project_training_page(request: Request, pid: str):
    """Training config + progress for a project."""
    from finetune_studio.webui.app import discovered_models, training_engine
    ctx = _project_ctx(pid)
    if not ctx:
        return RedirectResponse(url="/projects", status_code=302)
    return templates.TemplateResponse(
        request,
        "project_training.html",
        {**ctx, "models": discovered_models, "training_state": training_engine.state},
    )


@router.get("/projects/{pid}/testing", response_class=HTMLResponse)
async def project_testing_page(request: Request, pid: str):
    """Testing / inference playground for a project."""
    from finetune_studio.webui.app import discovered_models, inference_engine
    ctx = _project_ctx(pid)
    if not ctx:
        return RedirectResponse(url="/projects", status_code=302)
    return templates.TemplateResponse(
        request,
        "project_testing.html",
        {**ctx, "models": discovered_models, "inference_engine": inference_engine},
    )


@router.get("/projects/{pid}/models", response_class=HTMLResponse)
async def project_models_page(request: Request, pid: str):
    """Model browser for a project."""
    from finetune_studio.webui.app import discovered_models
    ctx = _project_ctx(pid)
    if not ctx:
        return RedirectResponse(url="/projects", status_code=302)
    return templates.TemplateResponse(
        request,
        "project_models.html",
        {**ctx, "models": discovered_models},
    )


# ── Other project-scoped pages ──────────────────────────────────────────

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
            "pid": pid,
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
    for run in runs:
        run["latest_benchmark"] = _latest_benchmark(run["id"])
    all_benchmarks = []
    run_name_map = {r["id"]: r["name"] for r in runs}
    import time as _time
    for run in runs:
        for b in db.list_benchmarks(run["id"]):
            b["_run_name"] = run_name_map.get(run["id"], run["id"])
            b["_ran_at_str"] = _time.strftime("%Y-%m-%d %H:%M", _time.localtime(b["ran_at"]))
            all_benchmarks.append(b)
    all_benchmarks.sort(key=lambda x: x["ran_at"], reverse=True)
    comparison_runs = [runs[0]["id"], runs[1]["id"]] if len(runs) >= 2 else []
    return templates.TemplateResponse(
        request,
        "benchmarks.html",
        {
            "request": request,
            "project": project,
            "pid": pid,
            "runs": runs,
            "suites": suites,
            "all_benchmarks": all_benchmarks,
            "comparison_runs": comparison_runs,
        },
    )


@router.get("/projects/{pid}/chat", response_class=HTMLResponse)
async def project_chat_page(request: Request, pid: str):
    """Deep-link to chat with a specific project pre-selected."""
    from finetune_studio import db
    project = db.get_project(pid)
    if not project:
        return RedirectResponse(url="/projects", status_code=302)
    return templates.TemplateResponse(request, "chat_v2.html", {"request": request, "project": project, "pid": pid})


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
            "pid": pid,
            "models": discovered_models,
        },
    )
