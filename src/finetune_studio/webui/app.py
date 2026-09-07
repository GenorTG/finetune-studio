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

from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles

from finetune_studio import db
from finetune_studio.config import settings
import os

from finetune_studio.models.registry import ModelInfo, scan_models
from finetune_studio.testing.inference import InferenceEngine
from finetune_studio.training.engine import TrainingEngine

training_engine = TrainingEngine()
inference_engine = InferenceEngine()
discovered_models: list[ModelInfo] = []

def _on_training_update(state):
    """Persist training progress to the DB row tagged on the engine."""
    rid = training_engine.current_run_id
    if not rid:
        return
    metrics = {
        "step": state.current_step,
        "total_steps": state.total_steps,
        "loss": state.loss,
        "learning_rate": state.learning_rate,
        "epoch": state.epoch,
        "elapsed": state.elapsed,
        "eta": state.eta,
    }
    fields: dict = {"status": state.status, "metrics": metrics}
    if state.status == "done":
        fields["finished_at"] = __import__("time").time()
        fields["output_path"] = str(Path(training_engine.config.output_dir) / "adapter")
    elif state.status == "error":
        fields["finished_at"] = __import__("time").time()
        fields["notes"] = (state.error or state.message)[:500]
    try:
        db.update_run(rid, **fields)
    except Exception:  # noqa: BLE001, S110
        pass

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
    # Init DB and hook training -> DB persistence.
    db.init_db()
    training_engine.on_update(_on_training_update)
    yield

app = FastAPI(title="Finetune Studio", version="0.1.0", lifespan=lifespan)

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
    agentic,
    benchmarks,
    chat_v2,
    comparison,
    data,
    data_editor,
    data_prep,
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
app.include_router(hf_models.router, prefix="/api", tags=["hf-models"])  # type: ignore[has-type]  # /api/hf/* + /api/shared-models/*
app.include_router(agentic.router, prefix="/api/agentic", tags=["agentic"])  # type: ignore[has-type]
app.include_router(system.router)  # type: ignore[has-type]  # /api/system/* — RAM/VRAM snapshot
app.include_router(data_prep.router, prefix="/api", tags=["data-prep"])  # type: ignore[has-type]
app.include_router(data_prep._pages)  # type: ignore[has-type]  # HTML page /projects/{pid}/data-prep
app.include_router(rag.router, prefix="/api/projects", tags=["rag"])  # type: ignore[has-type]  # /api/projects/{pid}/rag/*

