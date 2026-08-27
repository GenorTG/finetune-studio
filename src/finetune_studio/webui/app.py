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
from finetune_studio.models.registry import scan_models
from finetune_studio.testing.inference import InferenceEngine
from finetune_studio.training.engine import TrainingEngine

training_engine = TrainingEngine()
inference_engine = InferenceEngine()
from finetune_studio.models.registry import ModelInfo
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
    print("Scanning model directories...")
    discovered_models = scan_models(settings.model_dirs)
    print(f"Found {len(discovered_models)} models")
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

app.mount("/static", StaticFiles(directory=str(static_dir)), name="static")

from finetune_studio.webui.routes import (
    agentic,
    benchmarks,
    chat_v2,
    comparison,
    data,
    data_editor,
    models,
    pages,
    projects,
    quality,
    testing,
    training,
)

app.include_router(pages.router)  # type: ignore[has-type]
app.include_router(models.router, prefix="/api/models", tags=["models"])  # type: ignore[has-type]
app.include_router(training.router, prefix="/api/training", tags=["training"])  # type: ignore[has-type]
app.include_router(data.router, prefix="/api/data", tags=["data"])  # type: ignore[has-type]
app.include_router(testing.router, prefix="/api/testing", tags=["testing"])  # type: ignore[has-type]
app.include_router(comparison.router, prefix="/api/compare", tags=["compare"])  # type: ignore[has-type]
app.include_router(projects.router, prefix="/api/projects", tags=["projects"])  # type: ignore[has-type]
app.include_router(quality.router)  # type: ignore[has-type]  # already self-prefixed /api/data
app.include_router(benchmarks.router, prefix="/api/benchmarks", tags=["benchmarks"])  # type: ignore[has-type]
app.include_router(data_editor.router, prefix="/api/data-editor", tags=["data-editor"])  # type: ignore[has-type]
app.include_router(chat_v2.router, prefix="/api/chat-v2", tags=["chat-v2"])  # type: ignore[has-type]
app.include_router(agentic.router, prefix="/api/agentic", tags=["agentic"])  # type: ignore[has-type]
