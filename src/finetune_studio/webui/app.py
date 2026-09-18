"""Gradio web app composition.

WHAT THIS FILE DOES
==================
The main entry point for the web UI. Composes all the tabs
(data, models, training, testing, comparison) into a single Gradio
interface and launches it on port 7860.

KEY CONCEPTS
============
- Gradio: a Python library for creating web UIs for ML models.
  Defines UI as Python objects, no HTML/JS needed.
- Tab-based layout: each major feature gets its own tab.
- Event handlers: when the user clicks a button, we run a Python function.
- State management: the UI keeps state across interactions (which
  model is loaded, what test suite is selected, etc.).
"""

import json
import os
import time
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from finetune_studio import db
from finetune_studio.config import settings
from finetune_studio.models.registry import ModelInfo, scan_models
from finetune_studio.testing.inference import InferenceEngine
from finetune_studio.training.engine import TrainingEngine

training_engine = TrainingEngine()
inference_engine = InferenceEngine()
discovered_models: list[ModelInfo] = []

@asynccontextmanager
async def lifespan(app: FastAPI):
    global discovered_models
    # Ensure model directories exist
    for d in settings.model_dirs:
        os.makedirs(d, exist_ok=True)
    # Merge extra dirs from env/config
    dirs = list(settings.model_dirs)
    for d in settings.model_dirs_extra:
        if d not in dirs:
            dirs.append(d)
    print(f"Scanning {len(dirs)} model directories...")
    discovered_models = scan_models(dirs)
    print(f"Found {len(discovered_models)} models")
    for m in discovered_models:
        vision = " 👁 vision" if getattr(m, "vision", False) else ""
        print(f"  {m.name} ({m.format}, {m.size_gb}GB{vision})")
    # Init DB. Training runs bind their own DB persister when started
    # (training.run_persistence.attach_run).
    db.init_db()
    # Re-attach to in-flight HF downloads from the previous session so
    # the UI shows them (and we can mark the dead ones as cancelled).
    try:
        from finetune_studio.webui.routes.hf_models import restore_in_progress_downloads
        restore_in_progress_downloads()
    except Exception:  # noqa: BLE001
        pass
    # Finalize system_updates / training_runs orphaned when the previous
    # process was restarted (APPLY UPDATE kills its streaming worker; a hard
    # kill leaves training rows in loading/training/saving).
    try:
        n = db.reconcile_stale_updates()
        if n:
            print(f"Reconciled {n} stale system_updates row(s)")
        n_runs = db.reconcile_stale_runs()
        if n_runs:
            print(f"Reconciled {n_runs} stale training_runs row(s)")
        n = db.reconcile_stale_data_prep()
        if n:
            print(f"Reconciled {n} stale data_prep_runs row(s)")
        n = db.reconcile_stale_rag_builds()
        if n:
            print(f"Reconciled {n} stale rag_corpora row(s)")
        n = db.reconcile_stale_exports()
        if n:
            print(f"Reconciled {n} stale model_exports row(s)")
    except Exception:  # noqa: BLE001
        pass
    yield

app = FastAPI(title="Finetune Studio", version="0.1.0", lifespan=lifespan)


def _activity_kind(path: str) -> str:
    """Classify mutating API paths for the global operation feed."""
    p = path.lower()
    # Model load/unload/refresh must be checked BEFORE the broader inference
    # prefix so it does not steal those endpoints.
    if (
        "/models/load" in p
        or "/models/unload" in p
        or "/models/refresh" in p
        or "/inference/load" in p
        or "/inference/unload" in p
        or "/chat-v2/load" in p
        or "/chat-v2/unload" in p
        or "/providers/" in p
    ):
        return "model_load"
    # More specific checks so they win over the generic fallbacks below.
    if p.startswith(("/api/inference/", "/api/chat-v2/")) or "/compare/rag/chat" in p:
        return "inference"
    if "/rag/build" in p or "/rag/rebuild" in p or "/rag/rebuild-vectors" in p:
        return "rag_build"
    if "/rag/" in p and any(x in p for x in ("/chat", "/query", "/search")):
        return "rag_query"
    if "/upload" in p or "/promote" in p or "/reprocess" in p or p.endswith("/sources"):
        return "upload"
    if "/testing/" in p or p.endswith("/testing"):
        return "testing"
    if "/benchmark" in p or "/run-suite" in p:
        return "benchmark"
    if "/export" in p or p.endswith("/merge"):
        return "export"
    if "/runs" in p and p.endswith("/start"):
        return "training"
    if p == "/api/training/start" or "/training/" in p and p.endswith("/start"):
        return "training"
    if p.startswith("/api/hf/"):
        return "download"
    if "/chat" in p and "/data-prep" in p:
        return "data_prep"
    if "/data-prep/" in p:
        return "data_prep"
    if "/system-update" in p or "/system/" in p:
        return "system_update"
    return "operation"


@app.middleware("http")
async def record_activity_operations(request: Request, call_next):
    """Persist every mutating API operation after its response completes."""
    path = request.url.path
    if (
        request.method not in {"POST", "PUT", "PATCH", "DELETE"}
        or path.startswith("/api/activity")
        or path.endswith("/events")
    ):
        return await call_next(request)
    started = time.time()
    try:
        response = await call_next(request)
    except Exception:
        _record_activity_event(request, started, 500)
        raise
    _record_activity_event(request, started, response.status_code)
    return response


def _activity_summary(path: str, method: str, http_status: int) -> str:
    """Human one-liner for the activity feed (replaces 'POST /api/x → 200').

    Screen-scrapes the path for the resource the user actually acted on:
    filenames, run ids, dataset names. The project name is NOT baked in —
    it renders as its own badge from project_id.
    """
    try:
        # Keyed on the marker strings, not on path order.
        if "/data-prep/start" in path:
            return "Start Q&A mining"
        if "/data-prep/chat" in path:
            return "Agent mining question"
        if "/data-prep/qa/bulk" in path:
            return "Bulk-approve/reject Q&A"
        if "/data-prep/export" in path:
            return "Export Q&A dataset"
        if path.endswith("/data-prep/sources"):
            return "Promote file → source"
        if "/files/upload" in path:
            return "Upload + parse files"
        if "/training/start" in path:
            return "Start training run"
        if "/run-suite" in path:
            return "Run test suite"
        if any(x in path for x in ("/export-gguf", "/quantize", "/export")):
            return "Export model"
        if "/load" in path and "models" in path:
            return "Load model"
        if "/unload" in path:
            return "Unload model"
        if "/download/start" in path or path.startswith("/api/hf/"):
            return "HF download"
        if "/projects" in path and path.count("/") <= 3:
            return "Create project"
        verb = {
            "POST": "Create", "PUT": "Update", "PATCH": "Update",
            "DELETE": "Delete",
        }.get(method.upper(), "Operation")
        return verb
    except Exception:  # noqa: BLE001 — summary must never break recording
        return "Operation"


def _record_activity_event(request: Request, started: float, http_status: int) -> None:
    """Best-effort event write; logging must never break the API response."""
    try:
        parts = request.url.path.strip("/").split("/")
        project_id = parts[2] if len(parts) > 2 and parts[:2] == ["api", "projects"] else ""
        status = "done" if http_status < 400 else "error"
        db.record_activity_event(
            kind=_activity_kind(request.url.path),
            operation=request.url.path,
            method=request.method,
            path=request.url.path,
            project_id=project_id,
            status=status,
            http_status=http_status,
            message=_activity_summary(request.url.path, request.method, http_status),
            started_at=started,
            finished_at=time.time(),
        )
    except Exception:  # noqa: BLE001, S110 — activity logging must never break the API response
        pass

# ── CORS + Trusted Hosts (configured via Settings page) ──
def _apply_hosting_middleware():
    """Apply CORS and trusted-host middleware from user settings."""
    settings_path = Path.home() / ".finetune-studio" / "settings.json"
    user: dict = {}
    if settings_path.exists():
        try:
            user = json.loads(settings_path.read_text())
        except Exception:
            pass
    origins = [o for o in user.get("cors_origins", []) if o]
    if origins:
        app.add_middleware(
            CORSMiddleware,
            allow_origins=origins,
            allow_credentials=bool(user.get("cors_allow_credentials", True)),
            allow_methods=["*"],
            allow_headers=["*"],
        )
    # Proxy headers (for reverse-proxy deployments behind nginx/caddy)
    if user.get("proxy_headers"):
        from uvicorn.middleware.proxy_headers import ProxyHeadersMiddleware
        app.add_middleware(ProxyHeadersMiddleware, trusted_hosts="*")

_apply_hosting_middleware()

static_dir = Path(__file__).parent / "static"
templates_dir = Path(__file__).parent / "templates"
static_dir.mkdir(parents=True, exist_ok=True)
(static_dir / "css").mkdir(exist_ok=True)
(static_dir / "js").mkdir(exist_ok=True)

class _NoCacheStatic(StaticFiles):
    """Static files with no-cache headers for css/js so deploys apply instantly.

    Images/fonts keep default caching (file lookup is cheap).
    """

    async def get_response(self, path, scope):
        resp = await super().get_response(path, scope)
        # Cache-bust css/js so the modal-CSS fix and similar ship immediately.
        if path.endswith((".css", ".js")):
            resp.headers["Cache-Control"] = "no-cache, must-revalidate"
            resp.headers["Pragma"] = "no-cache"
            resp.headers["Expires"] = "0"
        return resp


app.mount("/static", _NoCacheStatic(directory=str(static_dir)), name="static")

from finetune_studio.webui.routes import (
    benchmarks,
    chat_v2,
    comparison,
    data,
    data_editor,
    data_prep,
    data_prep_chat,
    hf_models,
    models,
    pages,
    projects,
    quality,
    rag,
    system,
    testing,
    training,
)

app.include_router(pages.router)  # type: ignore[has-type]
app.include_router(models.router, prefix="/api/models", tags=["models"])  # type: ignore[has-type]
app.include_router(models.inference_router, prefix="/api/inference", tags=["inference"])  # type: ignore[has-type]
app.include_router(training.router, prefix="/api/training", tags=["training"])  # type: ignore[has-type]
app.include_router(data.router, prefix="/api/data", tags=["data"])  # type: ignore[has-type]
app.include_router(testing.router, prefix="/api/testing", tags=["testing"])  # type: ignore[has-type]
app.include_router(comparison.router, prefix="/api/compare", tags=["compare"])  # type: ignore[has-type]
app.include_router(projects.router, prefix="/api/projects", tags=["projects"])  # type: ignore[has-type]
app.include_router(quality.router)  # type: ignore[has-type]  # already self-prefixed /api/data
app.include_router(benchmarks.router, prefix="/api/benchmarks", tags=["benchmarks"])  # type: ignore[has-type]
app.include_router(data_editor.router, prefix="/api/data-editor", tags=["data-editor"])  # type: ignore[has-type]
app.include_router(chat_v2.router, prefix="/api/chat-v2", tags=["chat-v2"])  # type: ignore[has-type]
app.include_router(data_prep.router, prefix="/api", tags=["data-prep"])  # type: ignore[has-type]  # /api/projects/{pid}/data-prep/*
app.include_router(data_prep_chat.router, prefix="/api", tags=["data-prep-chat"])  # type: ignore[has-type]  # /api/projects/{pid}/data-prep/chat*
app.include_router(data_prep._pages)  # type: ignore[has-type]  # HTML page /projects/{pid}/data-prep
app.include_router(hf_models.router, prefix="/api")  # type: ignore[has-type]  # /api/hf/* + /api/shared-models/*
app.include_router(system.router)  # type: ignore[has-type]  # /api/system/* — RAM/VRAM snapshot
app.include_router(rag.router, prefix="/api/projects", tags=["rag"])  # type: ignore[has-type]  # /api/projects/{pid}/rag/*

from finetune_studio.webui.routes import exports as _exports

app.include_router(_exports.router, prefix="/api", tags=["exports"])  # type: ignore[has-type]  # /api/projects/{pid}/runs/{rid}/export

from finetune_studio.webui.routes import updates as _updates

app.include_router(_updates.router, prefix="/api", tags=["system-updates"])  # type: ignore[has-type]  # /api/system/update*

from finetune_studio.webui.routes import activity as _activity

app.include_router(_activity.router)  # type: ignore[has-type]  # /api/activity

from finetune_studio.webui.routes import settings as _settings

app.include_router(_settings.router)  # type: ignore[has-type]  # /api/settings

from finetune_studio.webui.routes import datasets as _datasets

app.include_router(_datasets.router, prefix="/api", tags=["datasets"])  # type: ignore[has-type]  # /api/projects/{pid}/datasets/*

# Stage 1 — File library (raw + converted dual storage, folders, dedup,
# versioning, trash). Routes self-prefix their full path.
from finetune_studio.webui.routes import file_library as _file_library

app.include_router(_file_library.router, prefix="/api", tags=["file-library"])  # type: ignore[has-type]  # /api/projects/{pid}/files/* + /folders/*

from finetune_studio.webui.routes import project_models as _project_models

app.include_router(_project_models.router, prefix="/api", tags=["project-models"])  # type: ignore[has-type]  # /api/projects/{pid}/models/.../contents

from finetune_studio.webui.routes import project_rag as _project_rag

app.include_router(_project_rag.router, prefix="/api", tags=["project-rag"])  # type: ignore[has-type]  # /api/projects/{pid}/rag/docs* + /rebuild

from finetune_studio.webui.routes import (
    project_settings as _project_settings,
)

app.include_router(_project_settings.router, prefix="/api", tags=["project-settings"])  # type: ignore[has-type]  # /api/projects/{pid}/logs
