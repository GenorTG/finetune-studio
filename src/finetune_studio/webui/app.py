"""FastAPI app composition.

Builds the ``app`` object served on port 7860: lifespan startup (model scan,
DB init, restart reconciliation), the mutating-request activity middleware,
CORS / proxy-header middleware from ``$FTS_ROOT/settings.json``, the
no-cache static mount, and every router include. Also owns the process-wide
``training_engine`` / ``inference_engine`` / ``discovered_models`` singletons
that route modules import lazily from here.
"""

import json
import logging
import os
import time
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from finetune_studio import (
    db,
    logsetup,
)
from finetune_studio.config import settings
from finetune_studio.models.manager import get_manager
from finetune_studio.models.registry import ModelInfo, scan_models
from finetune_studio.training.remote import RemoteTrainingEngine

logsetup.setup("web")          # $FTS_ROOT/logs/web.log (+ web-access.log), secrets redacted
_log = logging.getLogger(__name__)
# Training runs as a supervised job process (jobs.kinds.training); this is the web app's view of it.
training_engine = RemoteTrainingEngine()
# The ONE *persistent* InferenceEngine instance in this process. ModelManager
# owns it (via the .engine property) — every named-provider load (data-prep's
# helper, /api/providers/*) delegates to this exact object, so this name
# and ModelManager's internal engine can never independently hold a model
# at the same time. Some routes (benchmarks.py, projects.py) do construct a
# short-lived second InferenceEngine() for a one-shot judge/benchmark run —
# that's fine ONLY if they free this engine's VRAM first via
# llama_loader.unload_all_models() (see benchmarks.py's
# _unload_global_inference) and unload their temp engine in a finally block.
# A second engine constructed without unloading this one first silently
# risks mixed GPU/CPU offload (GH-AAA) from two models resident at once.
inference_engine = get_manager().engine
discovered_models: list[ModelInfo] = []

IDLE_REAP_INTERVAL = 60


async def _idle_reaper():
    """Once the app has sat idle past the timeout, free cached memory (and engines the timers missed)."""
    import asyncio

    from finetune_studio.testing.inference import idle_timeout, release_idle_memory

    last_activity = time.time()
    released = True
    while True:
        await asyncio.sleep(IDLE_REAP_INTERVAL)
        timeout = idle_timeout()
        training_busy = training_engine.state.status in ("training", "loading", "saving")
        if training_busy or inference_engine._busy or timeout <= 0:
            last_activity, released = time.time(), False
            continue
        if inference_engine.model is not None:
            last_activity, released = max(last_activity, inference_engine._last_used), False
            if time.time() - inference_engine._last_used >= timeout:
                inference_engine.unload()
            continue
        if not released and time.time() - last_activity >= timeout:
            await asyncio.to_thread(release_idle_memory)
            released = True

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
    # Init DB. A training job persists its own run row (jobs.kinds.training).
    db.init_db()
    # Re-attach to in-flight HF downloads from the previous session so
    # the UI shows them (and we can mark the dead ones as cancelled).
    try:
        from finetune_studio.webui.routes.hf_models import restore_in_progress_downloads
        restore_in_progress_downloads()
    except Exception:  # noqa: BLE001
        _log.exception("Failed to restore in-progress HF downloads")
    # Finalize system_updates / training_runs orphaned when the previous
    # process was restarted (APPLY UPDATE kills its streaming worker; a hard
    # kill leaves training rows in loading/training/saving).
    from finetune_studio.webui.restart_cause import restart_cause

    cause = restart_cause()
    try:
        from finetune_studio.jobs import store as job_store
        job_store.prune()                       # finished job folders older than a week (results are in the DB)
    except Exception:  # noqa: BLE001 — housekeeping must not block startup
        _log.exception("Failed to prune old job folders")
    try:
        n = db.reconcile_stale_updates()
        if n:
            print(f"Reconciled {n} stale system_updates row(s)")
        # A run whose job process is still alive (the web restarted, training did not) is left alone.
        n_runs = db.reconcile_stale_runs(cause, keep=training_engine.running_run_ids())
        if n_runs:
            print(f"Reconciled {n_runs} stale training_runs row(s)")
    except Exception:  # noqa: BLE001
        _log.exception("Failed to reconcile stale updates/training runs")
    # Independent reconciles first, so a failed data-prep resume below cannot
    # skip them.
    try:
        from finetune_studio.webui.brain_build_jobs import running_build_ids
        # A build whose job process is still alive (the web restarted, the build did not) is left alone.
        n = db.reconcile_stale_brain_builds("interrupted by service restart" + cause, keep=running_build_ids())
        if n:
            print(f"Reconciled {n} stale brain_builds row(s)")
        from finetune_studio.webui.export_jobs import running_export_ids
        # An export whose job process is still alive (the web restarted, the export did not) is left alone.
        n = db.reconcile_stale_exports("interrupted by service restart" + cause, keep=running_export_ids())
        if n:
            print(f"Reconciled {n} stale model_exports row(s)")
        from finetune_studio.webui.testing_jobs import running_benchmark_ids
        # A test run / judge / comparison whose job process is still alive is left alone.
        n = db.reconcile_stale_benchmarks(cause, keep=running_benchmark_ids())
        if n:
            print(f"Reconciled {n} stale test run/judge job(s)")
        from finetune_studio.testing.scoring import rescore_legacy_benchmarks

        n = rescore_legacy_benchmarks()
        if n:
            print(f"Rescored {n} legacy test run(s) with verdict sources")
    except Exception:  # noqa: BLE001
        _log.exception("Failed to reconcile stale RAG builds/exports")
    # Files & Knowledge: move projects off the old source manifests (one-time), then finish parses a restart cut off.
    try:
        from finetune_studio.knowledge.migrate import migrate_all

        n = migrate_all()
        if n:
            print(f"Moved {n} project(s) to the one-parse source layout")
    except Exception:  # noqa: BLE001 - startup recovery must never block boot
        _log.exception("Knowledge source migration failed")
    import threading

    from finetune_studio.knowledge.sources import resume_unfinished

    threading.Thread(target=resume_unfinished, name="resume-parses", daemon=True).start()
    # Mining runs in the mining worker job: files whose worker outlived the restart are left alone, the rest are
    # interrupted (Fill gaps continues them) and still-queued files get a worker.
    try:
        from finetune_studio.data.mining import recover_after_restart
        recover_after_restart()
    except Exception:  # noqa: BLE001 - startup recovery must never block boot
        _log.exception("Mining restart recovery failed")
    import asyncio

    from finetune_studio.jobs import watch
    from finetune_studio.webui import export_jobs
    watch.on_finished(export_jobs.KIND, export_jobs.on_finished)
    from finetune_studio.data.mining import scheduler as mining_scheduler
    from finetune_studio.webui import brain_build_jobs, download_jobs
    watch.on_finished(mining_scheduler.KIND, mining_scheduler.on_finished)
    watch.on_finished(download_jobs.KIND, download_jobs.on_finished)        # settle row + refresh model registry
    watch.on_finished(brain_build_jobs.KIND, brain_build_jobs.on_finished)  # fail a row its dead job left active
    reaper = asyncio.create_task(_idle_reaper())
    watcher = asyncio.create_task(watch.run_forever())
    try:
        yield
    finally:
        reaper.cancel()
        watcher.cancel()

app = FastAPI(title="Finetune Studio", version="0.1.0", lifespan=lifespan)

from finetune_studio.webui import errors as _errors

_errors.install(app)   # one error contract: real status + {"detail", "code"}; styled 404 pages


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
        or "/chat-v2/load" in p
        or "/providers/" in p
    ):
        return "model_load"
    # More specific checks so they win over the generic fallbacks below.
    if p.startswith(("/api/inference/", "/api/chat-v2/")):
        return "inference"
    if "/brain/build" in p or "/brain/quick" in p or "/brain/rebuild-vectors" in p:
        return "brain_build"
    if "/brain/" in p and any(x in p for x in ("/chat", "/search")):
        return "brain_query"
    if "/upload" in p or "/promote" in p or p.endswith("/sources"):
        return "upload"
    if "/testing/" in p or p.endswith("/testing"):
        return "testing"
    if "/benchmark" in p or "/run-suite" in p:
        return "benchmark"
    if "/export" in p or p.endswith("/merge"):
        return "export"
    if p.endswith("/start") and ("/runs" in p or "/training/" in p):
        return "training"
    if p.startswith("/api/hf/"):
        return "download"
    if "/chat" in p and "/data-prep" in p:
        return "data_prep"
    if "/data-prep/" in p or "/mining/" in p:
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
        _record_activity_event(request, started, 500, "internal error; see the web log")
        raise
    detail = await _activity_response_error(response) if response.status_code >= 400 else ""
    _record_activity_event(request, started, response.status_code, detail)
    return response


async def _activity_response_error(response) -> str:
    """Read a small JSON error body for the activity row, then replay it unchanged to the client."""
    if not response.headers.get("content-type", "").startswith("application/json"):
        return ""
    try:
        size = int(response.headers.get("content-length", "0"))
    except ValueError:
        return ""
    if size <= 0 or size > 64 * 1024:
        return ""
    iterator = getattr(response, "body_iterator", None)
    if iterator is None:
        body = getattr(response, "body", b"")
    else:
        chunks = [chunk async for chunk in iterator]
        response.body_iterator = _replay_activity_body(chunks)
        body = b"".join(chunk.encode() if isinstance(chunk, str) else chunk for chunk in chunks)
    try:
        payload = json.loads(body)
    except (TypeError, ValueError, UnicodeDecodeError):
        return ""
    return str(payload.get("detail") or "") if isinstance(payload, dict) else ""


async def _replay_activity_body(chunks):
    for chunk in chunks:
        yield chunk


def _activity_summary(path: str, method: str, http_status: int = 200, error_detail: str = "") -> str:
    """Human one-liner for the activity feed (replaces 'POST /api/x → 200').

    Screen-scrapes the path for the resource the user actually acted on:
    filenames, run ids, dataset names. The project name is NOT baked in —
    it renders as its own badge from project_id. Errors use the sanitized API reason.
    """
    if http_status >= 400:
        detail = " ".join(logsetup.redact(error_detail).split())
        return f"Failed ({http_status}): {detail[:180]}" if detail else f"Failed (HTTP {http_status})"
    try:
        # Keyed on the marker strings, not on path order.
        if "/mining/jobs" in path and path.endswith("/cancel"):
            return "Cancel a mining job"
        if "/mining/jobs" in path:
            return "Start mining (per file)"
        if "/mining/" in path and path.endswith("/exclude"):
            return "Exclude a data point"
        if "/mining/" in path and path.endswith("/reset"):
            return "Reset a mined file"
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
        if method.upper() == "POST" and path.rstrip("/") == "/api/projects":
            return "Create project"
        verb = {
            "POST": "Create", "PUT": "Update", "PATCH": "Update",
            "DELETE": "Delete",
        }.get(method.upper(), "Operation")
        return verb
    except Exception:  # noqa: BLE001 — summary must never break recording
        return "Operation"


def _record_activity_event(request: Request, started: float, http_status: int, error_detail: str = "") -> None:
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
            message=_activity_summary(
                request.url.path,
                request.method,
                http_status,
                error_detail or getattr(request.state, "activity_error", ""),
            ),
            started_at=started,
            finished_at=time.time(),
        )
    except Exception:  # noqa: BLE001, S110 — activity logging must never break the API response
        pass

# ── CORS + proxy headers (configured via Settings page) ──
def _apply_hosting_middleware() -> list[str]:
    """Apply CORS and proxy-header middleware from user settings; returns the CORS origins (trusted for writes)."""
    from finetune_studio.webui.routes.settings import SETTINGS_PATH as settings_path
    user: dict = {}
    if settings_path.exists():
        try:
            user = json.loads(settings_path.read_text())
        except (OSError, ValueError) as e:
            _log.warning("Ignoring unreadable %s: %s", settings_path, e)
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
    return origins

_cors_origins = _apply_hosting_middleware()

# Outermost: who may reach the app at all (password / API tokens / cross-site and DNS-rebinding guards).
from finetune_studio.access.middleware import AccessMiddleware

app.add_middleware(AccessMiddleware, trusted_origins=_cors_origins)
# Outermost of all: every request is logged, also the ones the access check refuses.
from finetune_studio.webui.access_log import AccessLogMiddleware

app.add_middleware(AccessLogMiddleware)

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
        if path.endswith((".css", ".js")):
            resp.headers["Cache-Control"] = "no-cache, must-revalidate"
            resp.headers["Pragma"] = "no-cache"
            resp.headers["Expires"] = "0"
        return resp


app.mount("/static", _NoCacheStatic(directory=str(static_dir)), name="static")

from finetune_studio.webui.routes import (
    benchmarks,
    brain,
    brain_documents,
    brain_export,
    chat_v2,
    comparison,
    data,
    data_editor,
    data_prep,
    data_prep_chat,
    data_prep_preference,
    guide_chat,
    hf_models,
    knowledge,
    mining,
    models,
    pages,
    projects,
    quality,
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
app.include_router(comparison.pages)  # type: ignore[has-type]  # HTML page /projects/{pid}/compare
app.include_router(projects.router, prefix="/api/projects", tags=["projects"])  # type: ignore[has-type]
app.include_router(quality.router)  # type: ignore[has-type]  # already self-prefixed /api/data
app.include_router(benchmarks.router, prefix="/api/benchmarks", tags=["benchmarks"])  # type: ignore[has-type]
app.include_router(data_editor.router, prefix="/api/data-editor", tags=["data-editor"])  # type: ignore[has-type]
app.include_router(chat_v2.router, prefix="/api/chat-v2", tags=["chat-v2"])  # type: ignore[has-type]
app.include_router(data_prep.router, prefix="/api", tags=["data-prep"])  # type: ignore[has-type]  # /api/projects/{pid}/data-prep/*
app.include_router(data_prep_chat.router, prefix="/api", tags=["data-prep-chat"])  # type: ignore[has-type]  # /api/projects/{pid}/data-prep/chat*
app.include_router(mining.router, tags=["mining"])  # type: ignore[has-type]  # /api/projects/{pid}/mining/*
app.include_router(data_prep_preference.router, prefix="/api", tags=["data-prep-preference"])  # type: ignore[has-type]  # /api/projects/{pid}/data-prep/preference*
app.include_router(guide_chat.router, prefix="/api", tags=["guide"])  # type: ignore[has-type]  # /api/guide/* (SSE chat + KB)
app.include_router(data_prep._pages)  # type: ignore[has-type]  # HTML page /projects/{pid}/data-prep
app.include_router(hf_models.router, prefix="/api")  # type: ignore[has-type]  # /api/hf/* + /api/shared-models/*
app.include_router(system.router)  # type: ignore[has-type]  # /api/system/* — RAM/VRAM snapshot
app.include_router(brain.router, prefix="/api/projects", tags=["brain"])  # type: ignore[has-type]  # /api/projects/{pid}/brain*
app.include_router(brain_documents.router, prefix="/api/projects", tags=["brain"])  # type: ignore[has-type]
app.include_router(brain_export.router, prefix="/api/projects", tags=["brain"])  # type: ignore[has-type]
app.include_router(knowledge.router, prefix="/api/projects", tags=["knowledge"])  # type: ignore[has-type]  # /{pid}/knowledge/search

from finetune_studio.webui.routes import exports as _exports

app.include_router(_exports.router, prefix="/api", tags=["exports"])  # type: ignore[has-type]  # /api/projects/{pid}/runs/{rid}/export

from finetune_studio.webui.routes import updates as _updates

app.include_router(_updates.router, prefix="/api", tags=["system-updates"])  # type: ignore[has-type]  # /api/system/update*

from finetune_studio.webui.routes import activity as _activity

app.include_router(_activity.router)  # type: ignore[has-type]  # /api/activity

from finetune_studio.webui.routes import versions as _versions

app.include_router(_versions.router, prefix="/api", tags=["versions"])  # /api/projects/{pid}/versions* + datasets/subset

from finetune_studio.webui.routes import settings as _settings

app.include_router(_settings.router)  # type: ignore[has-type]  # /api/settings
from finetune_studio.webui.routes import access as _access

app.include_router(_access.router)  # /login, /api/auth/*, /api/settings/security*

from finetune_studio.webui.routes import helper_settings as _helper_settings

app.include_router(_helper_settings.router)  # type: ignore[has-type]  # /api/settings/helper* (local GGUF or API provider)

from finetune_studio.webui.routes import vision_settings as _vision_settings

app.include_router(_vision_settings.router)  # /api/settings/vision* (Vision reader: off, local GGUF + mmproj, API)

from finetune_studio.webui.routes import compute_device as _compute_device

app.include_router(_compute_device.router)  # /api/system/compute-device

from finetune_studio.webui.routes import service as _service

app.include_router(_service.router)  # /api/service/*: supervisor status, events, logs, restart

from finetune_studio.webui.routes import datasets as _datasets

app.include_router(_datasets.router, prefix="/api", tags=["datasets"])  # type: ignore[has-type]  # /api/projects/{pid}/datasets/*

# Stage 1 — File library (raw + converted dual storage, folders, dedup,
# versioning, trash). Routes self-prefix their full path.
from finetune_studio.webui.routes import file_library as _file_library

app.include_router(_file_library.router, prefix="/api", tags=["file-library"])  # type: ignore[has-type]  # /api/projects/{pid}/files/* + /folders/*

from finetune_studio.webui.routes import project_models as _project_models

app.include_router(_project_models.router, prefix="/api", tags=["project-models"])  # type: ignore[has-type]  # /api/projects/{pid}/models/.../contents

from finetune_studio.webui.routes import (
    project_settings as _project_settings,
)

app.include_router(_project_settings.router, prefix="/api", tags=["project-settings"])  # type: ignore[has-type]  # /api/projects/{pid}/logs
