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

from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import HTMLResponse, RedirectResponse
from fastapi.templating import Jinja2Templates
from finetune_studio import __version__ as APP_VERSION

templates = Jinja2Templates(directory=str(Path(__file__).parent.parent / "templates"))
templates.env.globals["app_version"] = APP_VERSION

# Custom Jinja2 filters for project stats
def _sum_benchmarks(runs):
    """Sum total benchmark count across all runs."""
    return sum(len(r.get("benchmarks", [])) for r in runs)

templates.env.filters["sum_benchmarks"] = _sum_benchmarks

router = APIRouter()


# ── Helpers ──────────────────────────────────────────────────────────────

def _require_project(pid: str):
    """Return project dict or None (caller redirects to /projects)."""
    from finetune_studio import db
    return db.get_project(pid)


# Export subdirectories a training run may produce under its output_path.
_EXPORT_FORMATS = {
    "merged": "safetensors",
    "abliterated": "safetensors",
    "gguf": "gguf",
    "gptq": "gptq",
}


def _dir_size_gb(path: str) -> float:
    """Total size of a directory tree in GB, rounded to 2 decimals."""
    import os
    total = 0
    for root, _dirs, files in os.walk(path):
        for f in files:
            try:
                total += os.path.getsize(os.path.join(root, f))
            except OSError:
                continue
    return round(total / (1024 ** 3), 2)


def _scan_run_models(runs: list[dict]) -> list[dict]:
    """Find exported models on disk for a project's training runs.

    Each run writes its exports into subdirectories of its output_path
    (merged/, gguf/, gptq/, abliterated/). Only non-empty directories that
    exist are reported.
    """
    import os
    import time as _time

    models: list[dict] = []
    for run in runs:
        output_path = (run.get("output_path") or "").strip()
        if not output_path or not os.path.isdir(output_path):
            continue
        for subdir, fmt in _EXPORT_FORMATS.items():
            path = os.path.join(output_path, subdir)
            if not os.path.isdir(path):
                continue
            try:
                if not os.listdir(path):
                    continue
                created = os.path.getmtime(path)
            except OSError:
                continue
            models.append({
                "name": f"{run.get('name') or run.get('id', '')}/{subdir}",
                "format": fmt,
                "size_gb": _dir_size_gb(path),
                "run_id": run.get("id", ""),
                "run_name": run.get("name", ""),
                "path": path,
                "mtime": created,
                "created_at": _time.strftime(
                    "%Y-%m-%d %H:%M", _time.localtime(created)
                ),
            })
    return models


def _project_ctx(pid: str) -> dict:
    """Build common template context for project pages."""
    from finetune_studio import db
    project = db.get_project(pid)
    if not project:
        return {}
    project["rags"] = db.list_rags(pid)
    project["runs"] = db.list_runs(pid)
    project["datasets"] = db.list_datasets(pid)
    for run in project["runs"]:
        run["benchmarks"] = db.list_benchmarks(run["id"])
    project["models"] = _scan_run_models(project["runs"])
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
    from finetune_studio.webui.app import discovered_models, inference_engine
    loaded = None
    if inference_engine.model is not None:
        # Find matching discovered model for metadata
        loaded_path = inference_engine.model_path
        loaded = next(
            (m for m in discovered_models if m.path == loaded_path),
            None,
        )
        if loaded is None:
            # Model is loaded but not in discovered set (e.g. via /api/chat-v2/load)
            from finetune_studio.models.loader import load_model_info
            info = load_model_info(loaded_path) or {}
            from types import SimpleNamespace
            loaded = SimpleNamespace(
                name=info.get("name") or loaded_path.split("/")[-1],
                path=loaded_path,
                format=info.get("format", ""),
                size_gb=info.get("size_gb", 0.0),
                architecture=info.get("architecture", ""),
                vision=getattr(inference_engine, "vision", False),
            )
        else:
            loaded.vision = getattr(inference_engine, "vision", False)
    return templates.TemplateResponse(
        request,
        "inference.html",
        {"request": request, "models": discovered_models, "loaded": loaded},
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


@router.get("/models", response_class=HTMLResponse)
async def models_index(request: Request):
    """Local model library — all discovered models with categories."""
    from finetune_studio.webui.app import discovered_models
    from finetune_studio import db
    # Enrich with project names for trained exports
    projects = {p["id"]: p["name"] for p in db.list_projects()}
    return templates.TemplateResponse(
        request,
        "models_index.html",
        {"request": request, "models": discovered_models, "projects": projects},
    )


@router.get("/hf-models", response_class=HTMLResponse)
async def hf_models_alias(request: Request):
    """Alias for /models/explore — renders the same HF model browser."""
    from finetune_studio.webui.app import discovered_models
    return templates.TemplateResponse(
        request,
        "hf_models.html",
        {"request": request, "models": discovered_models},
    )


@router.get("/projects/{pid}/export", response_class=HTMLResponse)
async def export_page(pid: str, request: Request):
    """Model export page — choose format, quant, and export trained runs."""
    from finetune_studio import db
    project = db.get_project(pid)
    if not project:
        raise HTTPException(status_code=404, detail="Project not found")
    project["runs"] = db.list_runs(pid)
    project["models"] = _scan_run_models(project["runs"])
    return templates.TemplateResponse(
        request,
        "export_models.html",
        {"request": request, "project": project, "pid": pid},
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
    """Project overview dashboard — stats, recent runs/models/files, activity."""
    from finetune_studio.data.fs import file_library as fl
    from finetune_studio.webui.project_dashboard import build_dashboard_ctx

    ctx = _project_ctx(pid)
    if not ctx:
        return RedirectResponse(url="/projects", status_code=302)
    files = fl.list_files(pid)
    dash = build_dashboard_ctx(ctx["project"], pid, files=files)
    return templates.TemplateResponse(
        request,
        "project.html",
        {**ctx, **dash, "request": request},
    )


# ── Project-scoped pages ────────────────────────────────────────────────

@router.get("/projects/{pid}/data", response_class=HTMLResponse)
async def project_data_page(request: Request, pid: str):
    """File browser for a project (library + trash + upload)."""
    from finetune_studio.data.fs import file_library as fl
    from finetune_studio.webui.project_data_browser import build_file_browser_ctx

    ctx = _project_ctx(pid)
    if not ctx:
        return RedirectResponse(url="/projects", status_code=302)
    files = fl.list_files(pid, include_deleted=True)
    browser = build_file_browser_ctx(ctx["project"], files=files)
    return templates.TemplateResponse(
        request,
        "project_data.html",
        {**ctx, **browser, "request": request},
    )


@router.get("/projects/{pid}/training", response_class=HTMLResponse)
async def project_training_page(request: Request, pid: str):
    """Training config + progress for a project."""
    from finetune_studio import db
    from finetune_studio.webui.app import discovered_models, training_engine
    ctx = _project_ctx(pid)
    if not ctx:
        return RedirectResponse(url="/projects", status_code=302)
    # ?run=<id> opens the run-detail panel above the config card.
    detail_run = None
    run_id = request.query_params.get("run")
    if run_id:
        candidate = db.get_run(run_id)
        # Only show runs for this project (don't leak cross-project data).
        if candidate and candidate.get("project_id") == pid:
            detail_run = candidate
    return templates.TemplateResponse(
        request,
        "project_training.html",
        {
            **ctx,
            "models": discovered_models,
            "training_state": training_engine.state,
            "detail_run": detail_run,
            "detail_run_id": run_id or "",
        },
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
    from finetune_studio.webui.routes.benchmarks import (
        _discover_suites,
        _latest_benchmark,
    )
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
            b["_ran_at_str"] = _time.strftime(
                "%Y-%m-%d %H:%M", _time.localtime(b["ran_at"])
            )
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
    """Project chat page — chat with the project's production model, optionally
    augmented by any enabled RAG corpora attached to the project."""
    from finetune_studio import db
    project = db.get_project(pid)
    if not project:
        return RedirectResponse(url="/projects", status_code=302)
    rags = db.list_rags(pid)
    return templates.TemplateResponse(
        request,
        "chat_v2.html",
        {"request": request, "project": project, "pid": pid, "rags": rags},
    )


# ── Settings & Debug Info ───────────────────────────────────────────

@router.get("/settings", response_class=HTMLResponse)
async def settings_page(request: Request):
    """Settings, debug info, replay tutorial, system status."""
    from finetune_studio import __version__ as APP_VERSION
    return templates.TemplateResponse(
        request,
        "settings.html",
        {
            "request": request,
            "app_version": APP_VERSION,
        },
    )


@router.get("/api/debug/info")
async def debug_info():
    """Return system debug info for the Settings page."""
    import platform
    import sys
    import os
    from pathlib import Path
    from finetune_studio import __version__ as APP_VERSION

    info = {
        "app_version": APP_VERSION,
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "machine": platform.machine(),
        "hostname": platform.node(),
        "cwd": os.getcwd(),
        "user": os.environ.get("USER", "unknown"),
        "paths": {
            "data_dir": str(Path.home() / ".finetune-studio"),
            "hf_cache": str(Path.home() / ".cache" / "huggingface"),
            "shared_models": str(Path.home() / ".finetune-studio" / "shared_models"),
        },
    }

    # GPU info via nvidia-smi
    try:
        import subprocess
        r = subprocess.run(
            ["nvidia-smi", "--query-gpu=name,memory.total,memory.free,driver_version",
             "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=5,
        )
        if r.returncode == 0:
            gpus = []
            for line in r.stdout.strip().split("\n"):
                parts = [p.strip() for p in line.split(",")]
                if len(parts) >= 4:
                    gpus.append({
                        "name": parts[0],
                        "vram_total_mb": int(parts[1]),
                        "vram_free_mb": int(parts[2]),
                        "driver": parts[3],
                    })
            info["gpus"] = gpus
        else:
            info["gpus"] = []
    except Exception as e:
        info["gpus"] = []
        info["gpu_error"] = str(e)

    # Package versions
    pkgs = ["torch", "transformers", "peft", "llama_cpp", "sentence_transformers",
            "fastapi", "uvicorn", "jinja2", "playwright", "numpy", "pandas"]
    versions = {}
    for pkg in pkgs:
        try:
            mod = __import__(pkg)
            v = getattr(mod, "__version__", "?")
            versions[pkg] = v
        except ImportError:
            versions[pkg] = "(not installed)"
        except Exception as e:
            versions[pkg] = f"(error: {e})"
    info["packages"] = versions

    return info
