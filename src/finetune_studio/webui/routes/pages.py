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

from finetune_studio import __release_channel__ as RELEASE_CHANNEL
from finetune_studio import __version__ as APP_VERSION
from finetune_studio.webui.model_labels import model_label as _model_label

templates = Jinja2Templates(directory=str(Path(__file__).parent.parent / "templates"))
templates.env.globals["app_version"] = APP_VERSION
templates.env.globals["release_channel"] = RELEASE_CHANNEL

# Custom Jinja2 filters for project stats
def _sum_benchmarks(runs):
    """Sum total benchmark count across all runs."""
    return sum(len(r.get("benchmarks", [])) for r in runs)

templates.env.filters["sum_benchmarks"] = _sum_benchmarks
templates.env.filters["model_label"] = _model_label

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
            model_paths = [path]
            if fmt == "gguf":
                model_paths = [
                    os.path.join(path, name)
                    for name in sorted(os.listdir(path))
                    if name.endswith(".gguf")
                ]
                if not model_paths:
                    continue
            for model_path in model_paths:
                model_name = f"{run.get('name') or run.get('id', '')}/{subdir}"
                if fmt == "gguf":
                    model_name += f"/{os.path.basename(model_path)}"
                models.append({
                "name": model_name,
                "format": fmt,
                "size_gb": (
                    round(os.path.getsize(model_path) / (1024 ** 3), 2)
                    if os.path.isfile(model_path)
                    else _dir_size_gb(path)
                ),
                "run_id": run.get("id", ""),
                "run_name": run.get("name", ""),
                "path": model_path,
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
    from finetune_studio.models.registry import models_for_selectors
    from finetune_studio.webui.app import discovered_models, inference_engine
    selector_models = models_for_selectors(discovered_models)
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
        {"request": request, "models": selector_models, "loaded": loaded},
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
    from finetune_studio import db
    from finetune_studio.webui.app import discovered_models
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
    """Model export page — choose format, quant, and browse trained exports."""
    from finetune_studio import db
    from finetune_studio.training.export_capabilities import (
        probe_export_capabilities,
    )
    from finetune_studio.webui.routes.project_export import (
        annotate_runs_for_export,
        runs_by_id,
    )

    project = db.get_project(pid)
    if not project:
        raise HTTPException(status_code=404, detail="Project not found")
    project["runs"] = annotate_runs_for_export(db.list_runs(pid))
    project["models"] = _scan_run_models(project["runs"])
    caps = probe_export_capabilities()
    return templates.TemplateResponse(
        request,
        "export_models.html",
        {
            "request": request,
            "project": project,
            "pid": pid,
            "runs_by_id": runs_by_id(project["runs"]),
            "export_caps": caps.as_dict(),
        },
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

@router.get("/projects/{pid}/overview", response_class=HTMLResponse)
async def project_overview_alias(request: Request, pid: str):
    """Alias used by the sticky breadcrumb (QABUG-009)."""
    return RedirectResponse(url=f"/projects/{pid}", status_code=302)


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
    from finetune_studio.models.registry import models_for_training
    from finetune_studio.webui.app import discovered_models, training_engine
    from finetune_studio.webui.project_dashboard import resolve_production_run
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
    production = resolve_production_run(
        ctx["project"], ctx["project"].get("runs") or [],
    )
    return templates.TemplateResponse(
        request,
        "project_training.html",
        {
            **ctx,
            "models": models_for_training(discovered_models),
            "training_state": training_engine.state,
            "detail_run": detail_run,
            "detail_run_id": run_id or "",
            "production_run": production,
        },
    )


def _recent_suite_runs(pid: str, limit: int = 5) -> list[dict]:
    """Return the most recent benchmark suite runs for a project (newest first)."""
    import json
    import time as _time

    from finetune_studio import db

    runs = db.list_runs(pid)
    run_name_map = {r["id"]: r["name"] for r in runs}
    rows: list[dict] = []
    for run in runs:
        for b in db.list_benchmarks(run["id"]):
            scores = b.get("scores") or {}
            if isinstance(scores, str):
                try:
                    scores = json.loads(scores)
                except json.JSONDecodeError:
                    scores = {}
            pass_rate = scores.get("pass_rate") if isinstance(scores, dict) else None
            rows.append(
                {
                    "suite_name": b.get("suite_name") or b.get("suite") or "—",
                    "run_name": run_name_map.get(run["id"], run["id"]),
                    "pass_rate": pass_rate,
                    "ran_at": b.get("ran_at") or 0,
                    "ran_at_str": _time.strftime(
                        "%Y-%m-%d %H:%M",
                        _time.localtime(b.get("ran_at") or 0),
                    ),
                }
            )
    rows.sort(key=lambda x: x["ran_at"], reverse=True)
    return rows[:limit]


@router.get("/projects/{pid}/testing", response_class=HTMLResponse)
async def project_testing_page(request: Request, pid: str):
    """Testing / inference playground for a project."""
    from finetune_studio.webui.app import inference_engine
    from finetune_studio.webui.routes.benchmarks import _discover_suites
    from finetune_studio.webui.testing_models import (
        default_model_path_for_testing,
        models_for_testing_page,
    )

    ctx = _project_ctx(pid)
    if not ctx:
        return RedirectResponse(url="/projects", status_code=302)
    suites = _discover_suites(pid)
    recent_runs = _recent_suite_runs(pid, limit=5)
    # Project-scoped exports only (not global HF discovery) so merge-at-export
    # results appear in the selector and match auto-load.
    models = models_for_testing_page(ctx["project"].get("models") or [])
    default_path = default_model_path_for_testing(models)
    from finetune_studio.db import datasets as datasets_db
    from finetune_studio.models.helper import (
        DEFAULT_HELPER_LABEL,
        DEFAULT_HELPER_PROVIDER_ID,
        get_configured_helper_provider,
    )
    helper = get_configured_helper_provider()
    training_datasets = datasets_db.list_datasets(pid)
    return templates.TemplateResponse(
        request,
        "project_testing.html",
        {
            **ctx,
            "models": models,
            "default_model_path": default_path,
            "inference_engine": inference_engine,
            "suites": suites,
            "recent_suite_runs": recent_runs,
            "training_datasets": training_datasets,
            "helper_label": (helper or {}).get("label") or DEFAULT_HELPER_LABEL,
            "helper_provider_id": DEFAULT_HELPER_PROVIDER_ID,
        },
    )


@router.get("/projects/{pid}/models", response_class=HTMLResponse)
async def project_models_page(request: Request, pid: str):
    """Model browser for a project — trained exports with expand-row detail."""
    from finetune_studio.webui.app import discovered_models
    ctx = _project_ctx(pid)
    if not ctx:
        return RedirectResponse(url="/projects", status_code=302)
    runs_by_id = {r["id"]: r for r in ctx["project"].get("runs", [])}
    return templates.TemplateResponse(
        request,
        "project_models.html",
        {**ctx, "models": discovered_models, "runs_by_id": runs_by_id},
    )


@router.get("/projects/{pid}/rag", response_class=HTMLResponse)
async def project_rag_page(request: Request, pid: str):
    """RAG page — corpus build/chat plus docs-indexed inventory panel."""
    from finetune_studio.webui.routes.project_rag import (
        list_indexed_docs,
        total_chunk_count,
    )

    ctx = _project_ctx(pid)
    if not ctx:
        return RedirectResponse(url="/projects", status_code=302)
    indexed_docs = list_indexed_docs(pid)
    return templates.TemplateResponse(
        request,
        "rag.html",
        {
            **ctx,
            "indexed_docs": indexed_docs,
            "indexed_doc_count": len(indexed_docs),
            "indexed_chunk_count": total_chunk_count(indexed_docs),
        },
    )


# ── Other project-scoped pages ──────────────────────────────────────────

@router.get("/projects/{pid}/data/{dataset_path:path}", response_class=HTMLResponse)
async def data_editor_page(request: Request, pid: str, dataset_path: str):
    """Project-scoped data editor for a JSONL dataset."""
    from finetune_studio import db as _db
    project = _db.get_project(pid)
    if not project:
        return RedirectResponse(url="/projects", status_code=302)
    filename = Path(dataset_path).name or dataset_path
    return templates.TemplateResponse(
        request,
        "data_editor.html",
        {
            "request": request,
            "project": project,
            "pid": pid,
            "dataset": dataset_path,
            "filename": filename,
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
    runs = [
        r for r in db.list_runs(pid)
        if r.get("name") != "__base_model__"
    ]
    suites = _discover_suites(pid)
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
    done_runs = [
        r for r in runs
        if (r.get("status") or "").lower() in ("done", "completed")
    ]
    # runs are newest-first: A = oldest done (baseline), B = most-recent done
    cmp_default_a = ""
    cmp_default_b = ""
    if len(done_runs) >= 2:
        cmp_default_a = done_runs[-1]["id"]
        cmp_default_b = done_runs[0]["id"]
    elif len(done_runs) == 1:
        cmp_default_a = done_runs[0]["id"]
        cmp_default_b = next(
            (r["id"] for r in runs if r["id"] != cmp_default_a),
            "",
        )
    elif len(comparison_runs) == 2:
        cmp_default_a, cmp_default_b = comparison_runs[0], comparison_runs[1]

    # Latest benchmark detail for the per-case table (QABUG-012).
    latest_cases: list = []
    latest_scores: dict = {}
    latest_bench = all_benchmarks[0] if all_benchmarks else None
    latest_rid = ""
    if latest_bench:
        latest_cases = db.list_cases(latest_bench["id"])
        for c in latest_cases:
            c["name"] = c.get("case_name") or c.get("name") or ""
        raw_scores = latest_bench.get("scores") or latest_bench.get("scores_json") or {}
        if isinstance(raw_scores, str):
            import json as _json
            try:
                raw_scores = _json.loads(raw_scores)
            except _json.JSONDecodeError:
                raw_scores = {}
        latest_scores = raw_scores if isinstance(raw_scores, dict) else {}
        latest_rid = latest_bench.get("run_id") or ""

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
            "cmp_default_a": cmp_default_a,
            "cmp_default_b": cmp_default_b,
            "cases": latest_cases,
            "scores": latest_scores,
            "latest_run_id": latest_rid,
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


@router.get("/projects/{pid}/settings", response_class=HTMLResponse)
async def project_settings_page(request: Request, pid: str):
    """Project settings + WebUI log tail (no SSH needed for uvicorn.log)."""
    ctx = _project_ctx(pid)
    if not ctx:
        return RedirectResponse(url="/projects", status_code=302)
    return templates.TemplateResponse(
        request,
        "project_settings.html",
        {**ctx, "request": request},
    )


# ── Settings & Debug Info ───────────────────────────────────────────

@router.get("/settings", response_class=HTMLResponse)
async def settings_page(request: Request):
    """Settings, debug info, replay tutorial, system status."""
    from finetune_studio import __release_channel__ as RELEASE_CHANNEL
    from finetune_studio import __version__ as APP_VERSION
    return templates.TemplateResponse(
        request,
        "settings.html",
        {
            "request": request,
            "app_version": APP_VERSION,
            "release_channel": RELEASE_CHANNEL,
        },
    )


@router.get("/api/debug/info")
async def debug_info():
    """Return system debug info for the Settings page."""
    import os
    import platform
    import sys
    from pathlib import Path

    from finetune_studio import __release_channel__ as RELEASE_CHANNEL
    from finetune_studio import __version__ as APP_VERSION

    info = {
        "app_version": APP_VERSION,
        "release_channel": RELEASE_CHANNEL,
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
    import subprocess
    try:
        r = subprocess.run(  # noqa: ASYNC221  # sync probe; debug endpoint is best-effort
            ["nvidia-smi", "--query-gpu=name,memory.total,memory.free,driver_version",
             "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=5, check=False,
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
    except (FileNotFoundError, subprocess.SubprocessError, OSError, ValueError) as e:
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
        except (OSError, RuntimeError, AttributeError) as e:
            versions[pkg] = f"(error: {e})"
    info["packages"] = versions

    return info
