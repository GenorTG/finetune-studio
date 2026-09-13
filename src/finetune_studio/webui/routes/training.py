"""Training tab — start/stop training, monitor progress."""

from fastapi import APIRouter, Request
from fastapi.responses import PlainTextResponse, StreamingResponse

from finetune_studio.training.data import load_jsonl
from finetune_studio.training.engine import TrainingConfig
from finetune_studio.training.monitor import training_events
from finetune_studio.webui.app import training_engine
from pathlib import Path

router = APIRouter()

# ── Training Presets ──────────────────────────────────────────
# Real configurations for different model sizes and use cases.
# These are NOT toy configs — they're based on LoRA best practices
# from the Unsloth/PEFT documentation and fine-tuning guides.
TRAINING_PRESETS: dict[str, dict] = {
    # ── Small models (0.5B–1.5B) ──────────────────────────────
    "nano-fast": {
        "name": "Nano — Fast iteration",
        "description": "Quick runs for data validation. ~5–15 min on 0.6B.",
        "target_models": ["0.6B", "1.5B"],
        "min_vram_gb": 6,
        "lora_rank": 32,
        "lora_alpha": 64,
        "learning_rate": 2e-4,
        "num_epochs": 3,
        "batch_size": 4,
        "gradient_accumulation_steps": 2,
        "max_seq_length": 1024,
        "warmup_steps": 10,
        "weight_decay": 0.01,
        "save_steps": 50,
        "logging_steps": 5,
        "bf16": True,
        "unsloth": True,
        "merge_on_save": False,
    },
    "nano-quality": {
        "name": "Nano — Quality",
        "description": "Better convergence for small models. ~30–60 min.",
        "target_models": ["0.6B", "1.5B"],
        "min_vram_gb": 8,
        "lora_rank": 64,
        "lora_alpha": 128,
        "learning_rate": 1e-4,
        "num_epochs": 5,
        "batch_size": 2,
        "gradient_accumulation_steps": 4,
        "max_seq_length": 2048,
        "warmup_steps": 20,
        "weight_decay": 0.01,
        "save_steps": 100,
        "logging_steps": 10,
        "bf16": True,
        "unsloth": True,
        "merge_on_save": True,
    },
    # ── Medium models (3B–8B) ─────────────────────────────────
    "standard": {
        "name": "Standard — Balanced",
        "description": "Good default for 3B–8B models. ~2–4 hours on 7B.",
        "target_models": ["3B", "7B", "8B"],
        "min_vram_gb": 16,
        "lora_rank": 64,
        "lora_alpha": 128,
        "learning_rate": 2e-4,
        "num_epochs": 3,
        "batch_size": 2,
        "gradient_accumulation_steps": 4,
        "max_seq_length": 2048,
        "warmup_steps": 30,
        "weight_decay": 0.01,
        "save_steps": 100,
        "logging_steps": 10,
        "bf16": True,
        "unsloth": True,
        "merge_on_save": True,
    },
    "standard-long": {
        "name": "Standard — Long training",
        "description": "More epochs for better convergence. ~6–10 hours on 7B.",
        "target_models": ["3B", "7B", "8B"],
        "min_vram_gb": 16,
        "lora_rank": 64,
        "lora_alpha": 128,
        "learning_rate": 1e-4,
        "num_epochs": 6,
        "batch_size": 2,
        "gradient_accumulation_steps": 4,
        "max_seq_length": 2048,
        "warmup_steps": 50,
        "weight_decay": 0.01,
        "save_steps": 100,
        "logging_steps": 10,
        "bf16": True,
        "unsloth": True,
        "merge_on_save": True,
    },
    "high-rank": {
        "name": "High Rank — Maximum capacity",
        "description": "Larger adapter for complex tasks. ~4–8 hours on 7B.",
        "target_models": ["3B", "7B", "8B"],
        "min_vram_gb": 20,
        "lora_rank": 128,
        "lora_alpha": 256,
        "learning_rate": 1e-4,
        "num_epochs": 4,
        "batch_size": 1,
        "gradient_accumulation_steps": 8,
        "max_seq_length": 2048,
        "warmup_steps": 40,
        "weight_decay": 0.01,
        "save_steps": 100,
        "logging_steps": 10,
        "bf16": True,
        "unsloth": True,
        "merge_on_save": True,
    },
    # ── Large models (14B–27B) ────────────────────────────────
    "large": {
        "name": "Large — 14B/27B",
        "description": "For big models. ~8–20 hours on 14B.",
        "target_models": ["14B", "27B"],
        "min_vram_gb": 24,
        "lora_rank": 64,
        "lora_alpha": 128,
        "learning_rate": 1e-4,
        "num_epochs": 3,
        "batch_size": 1,
        "gradient_accumulation_steps": 8,
        "max_seq_length": 2048,
        "warmup_steps": 50,
        "weight_decay": 0.01,
        "save_steps": 100,
        "logging_steps": 10,
        "bf16": True,
        "unsloth": True,
        "merge_on_save": True,
    },
    # ── QLoRA (4-bit base) ────────────────────────────────────
    "qlora": {
        "name": "QLoRA — Memory efficient",
        "description": "4-bit base + LoRA. Fits 7B in ~6GB VRAM.",
        "target_models": ["3B", "7B", "8B"],
        "min_vram_gb": 6,
        "lora_rank": 64,
        "lora_alpha": 128,
        "learning_rate": 2e-4,
        "num_epochs": 3,
        "batch_size": 2,
        "gradient_accumulation_steps": 4,
        "max_seq_length": 2048,
        "warmup_steps": 30,
        "weight_decay": 0.01,
        "save_steps": 100,
        "logging_steps": 10,
        "bf16": True,
        "unsloth": True,
        "merge_on_save": True,
    },
}


def _get_presets() -> list[dict]:
    """Return all presets with their IDs."""
    return [{"id": k, **v} for k, v in TRAINING_PRESETS.items()]


def _get_preset(preset_id: str) -> dict | None:
    """Return a specific preset by ID."""
    p = TRAINING_PRESETS.get(preset_id)
    if p is None:
        return None
    return {"id": preset_id, **p}


def _apply_preset(preset_id: str, overrides: dict | None = None) -> TrainingConfig:
    """Build a TrainingConfig from a preset, with optional field overrides."""
    p = TRAINING_PRESETS.get(preset_id)
    if p is None:
        raise ValueError(f"Unknown preset: {preset_id}")
    kwargs = {k: v for k, v in p.items() if k not in ("name", "description", "target_models", "min_vram_gb")}
    if overrides:
        kwargs.update(overrides)
    return TrainingConfig(**kwargs)


@router.get("/presets")
async def list_presets():
    """Return all training presets."""
    return _get_presets()


@router.get("/presets/{preset_id}")
async def get_preset(preset_id: str):
    """Return a specific preset."""
    p = _get_preset(preset_id)
    if p is None:
        from fastapi import HTTPException
        raise HTTPException(status_code=404, detail=f"Unknown preset: {preset_id}")
    return p

@router.get("/status")
async def status():
    s = training_engine.state
    return {"status": s.status, "step": s.current_step, "total_steps": s.total_steps,
            "loss": s.loss, "learning_rate": s.learning_rate, "epoch": s.epoch,
            "elapsed": s.elapsed, "eta": s.eta, "message": s.message, "error": s.error}

@router.get("/status-text")
async def status_text():
    s = training_engine.state
    if s.status == "training":
        text = f"Training ({s.current_step}/{s.total_steps})"
    elif s.status == "loading":
        text = "Loading model…"
    elif s.status == "saving":
        text = "Saving…"
    elif s.status == "done":
        text = "Done"
    elif s.status == "error":
        text = "Error"
    else:
        text = "Idle"
    return PlainTextResponse(text)

@router.get("/progress")
async def progress():
    return StreamingResponse(training_events(training_engine), media_type="text/event-stream")


@router.get("/progress-text")
async def progress_text():
    """Plain-text one-shot progress string for dashboard polling.

    Returns something short the dashboard can render without streaming:
      "Idle"            — no run
      "step 124/1000"   — running
      "step 1000/1000 · loss 1.42" — running w/ loss
      "Saving…"         — saving
      "Done"            — terminal state
      "Error"           — terminal error
    """
    s = training_engine.state
    if s.status == "training":
        loss = f" · loss {s.loss:.2f}" if s.loss else ""
        return PlainTextResponse(f"step {s.current_step}/{s.total_steps}{loss}")
    if s.status == "loading":
        return PlainTextResponse("Loading model…")
    if s.status == "saving":
        return PlainTextResponse("Saving…")
    if s.status == "done":
        return PlainTextResponse(f"Done · {s.total_steps} steps")
    if s.status == "error":
        return PlainTextResponse(f"Error · {s.error or 'unknown'}")
    return PlainTextResponse("Idle")

@router.get("/runs")
async def list_training_runs():
    """List ALL training runs (across all projects)."""
    from finetune_studio.db.runs import list_runs
    return list_runs()


@router.get("/runs/{pid}")
async def list_training_runs_for_project(pid: str):
    """List training runs for a specific project."""
    from finetune_studio.db.runs import list_runs
    return list_runs(pid)


@router.post("/start")
async def start_training(request: Request):
    from finetune_studio import db
    body = await request.json()
    merge_on_save = bool(body.get("merge_on_save", True))
    preset_id = body.get("preset_id")
    overrides = body.get("overrides", {})
    if preset_id:
        try:
            config = _apply_preset(preset_id, overrides)
        except ValueError as e:
            return {"error": str(e)}
        # Preset doesn't carry model_path — apply from body after preset merge
        if not config.model_path and body.get("model_path"):
            config.model_path = body["model_path"]
    else:
        config = TrainingConfig(
            model_path=body.get("model_path", ""),
            output_dir=body.get("output_dir", "output"),
            lora_rank=int(body.get("lora_rank", 64)),
            lora_alpha=int(body.get("lora_alpha", 128)),
            learning_rate=float(body.get("learning_rate", 2e-4)),
            num_epochs=int(body.get("num_epochs", 3)),
            batch_size=int(body.get("batch_size", 2)),
            gradient_accumulation_steps=int(body.get("gradient_accumulation_steps", 4)),
            max_seq_length=int(body.get("max_seq_length", 2048)),
            warmup_steps=int(body.get("warmup_steps", 30)),
            weight_decay=float(body.get("weight_decay", 0.01)),
            save_steps=int(body.get("save_steps", 100)),
            logging_steps=int(body.get("logging_steps", 10)),
            bf16=bool(body.get("bf16", True)),
            unsloth=bool(body.get("unsloth", True)),
            merge_on_save=merge_on_save,
            export_gguf=bool(body.get("export_gguf", False)),
            gguf_quants=body.get("gguf_quants", ["f16", "q8_0", "q4_k_m", "q5_k_m"]),
        )
    data_path = body.get("data_path", "")
    dataset_id = body.get("dataset_id", "")
    project_id = body.get("project_id", "")
    if dataset_id and not data_path:
        # Resolve dataset_id → data_path so the rest of the pipeline stays path-based.
        from finetune_studio import db
        ds = db.get_dataset(dataset_id)
        if not ds or ds.get("project_id") != project_id:
            return {"error": f"dataset {dataset_id!r} not found in this project"}
        data_path = ds["data_path"]
        # Track that the dataset was used (for "last used" sorting in the UI).
        try:
            db.update_dataset(dataset_id, last_used_at=__import__("time").time())
        except Exception:
            pass
    if not data_path:
        return {"error": "No data_path / dataset_id provided"}
    if not config.model_path:
        return {"error": "No model_path provided"}
    training_data = load_jsonl(data_path)
    system_prompt = body.get("system_prompt", "")

    # Create a run record so it shows up in the project's "Past runs" list.
    # Also lets the activity feed's "GO TO →" deep-link to the right page.
    import time as _time
    run = db.create_run(
        project_id=project_id,
        name=f"Run · {Path(data_path).name}",
        base_model=config.model_path,
        data_path=data_path,
        rag_ids=[],
        settings_obj={
            "lora_rank": config.lora_rank,
            "learning_rate": config.learning_rate,
            "num_epochs": config.num_epochs,
            "batch_size": config.batch_size,
            "max_seq_length": config.max_seq_length,
            "merge_on_save": merge_on_save,
            "system_prompt": system_prompt,
        },
        system_prompt=system_prompt,
    )
    run_id = run["id"]
    run_started_at = _time.time()  # wall-clock for this run; written on first transition
    # Encode project_id into current_run_id so the activity feed can
    # derive the URL without needing an extra DB lookup.
    training_engine.current_run_id = f"{project_id}-{run_id}" if project_id else run_id
    training_engine.current_project_id = project_id
    training_engine.current_db_run_id = run_id

    # Push state changes (loss, status, final_loss) into the run row.
    # Captures started_at on the first training-state transition and
    # finished_at (+ duration) on every terminal state so the past-runs
    # table can render human time without re-computing from the engine.
    state_started_logged = {"value": False}  # closure-shared flag
    def _on_state_change(state):
        update: dict = {}
        # First transition into an active training state → stamp started_at.
        active_states = ("running", "training", "loading", "saving")
        terminal_states = ("done", "error")
        if state.status in active_states and not state_started_logged["value"]:
            state_started_logged["value"] = True
            update["started_at"] = run_started_at
            update["status"] = state.status
        elif state.status in active_states:
            update["status"] = state.status
        if state.status == "done":
            update["status"] = "done"
            finished = _time.time()
            update["finished_at"] = finished
            update["duration"] = max(0.0, finished - run_started_at)
            if state.loss:
                update["final_loss"] = state.loss
            metrics = {
                "total_steps": state.total_steps,
                "current_step": state.current_step,
                "loss": state.loss,
                "epoch": state.epoch,
                "elapsed": state.elapsed,
            }
            update["metrics"] = metrics
        elif state.status == "saving":
            update["status"] = "saving"
        elif state.status == "error":
            update["status"] = "failed"
            finished = _time.time()
            update["finished_at"] = finished
            update["duration"] = max(0.0, finished - run_started_at)
            update["error"] = state.error or state.message or "training failed"
        if not update:
            return
        try:
            db.update_run(run_id, **update)
        except Exception:
            pass

    training_engine.on_update(_on_state_change)

    training_engine.start(config, training_data, system_prompt)
    return {"status": "started", "steps": training_engine.state.total_steps, "run_id": run_id}

@router.post("/stop")
async def stop_training():
    training_engine.stop()
    return {"status": "stopping"}


@router.post("/runs/{run_id}/export")
async def export_run(run_id: str, request: Request):
    """Export a trained run to GGUF (standalone, post-training).

    Body:
        quants: list of quant types to export (default: ["f16", "q8_0", "q4_k_m", "q5_k_m"])
        force: overwrite existing exports (default: false)
    """
    from finetune_studio import db
    body = await request.json()
    quants = body.get("quants", ["f16", "q8_0", "q4_k_m", "q5_k_m"])
    force = bool(body.get("force", False))

    run = db.get_run(run_id)
    if not run:
        return {"error": "run not found"}

    output_path = (run.get("output_path") or "").strip()
    if not output_path:
        return {"error": "run has no output_path"}

    merged_dir = os.path.join(output_path, "merged")
    if not os.path.isdir(merged_dir) or not os.listdir(merged_dir):
        return {"error": "no merged model to export — run merge first"}

    # Check if already exported
    gguf_dir = os.path.join(output_path, "gguf")
    if os.path.isdir(gguf_dir) and os.listdir(gguf_dir) and not force:
        return {
            "ok": True,
            "status": "skipped",
            "gguf_path": gguf_dir,
            "message": "GGUF already exists. Use force=true to overwrite.",
        }

    # Build a temp config with the requested quants
    from finetune_studio.training.engine import TrainingConfig
    cfg = TrainingConfig(output_dir=output_path, gguf_quants=quants)
    from finetune_studio.training.engine import TrainingEngine
    engine = TrainingEngine()
    engine.config = cfg
    result = engine._do_export_gguf(output_path)
    return {"ok": True, "status": "exported" if not result.get("skipped") else "skipped", **result}


@router.get("/runs/{run_id}/exports")
async def list_exports(run_id: str):
    """List all exports (merged, gguf, adapter) for a training run."""
    from finetune_studio import db
    run = db.get_run(run_id)
    if not run:
        return {"error": "run not found"}

    output_path = (run.get("output_path") or "").strip()
    exports = {"run_id": run_id, "output_path": output_path, "formats": {}}

    if output_path:
        merged_dir = os.path.join(output_path, "merged")
        if os.path.isdir(merged_dir) and os.listdir(merged_dir):
            exports["formats"]["merged"] = {"path": merged_dir, "files": os.listdir(merged_dir)}

        adapter_dir = os.path.join(output_path, "adapter")
        if os.path.isdir(adapter_dir) and os.listdir(adapter_dir):
            exports["formats"]["adapter"] = {"path": adapter_dir, "files": os.listdir(adapter_dir)}

        gguf_dir = os.path.join(output_path, "gguf")
        if os.path.isdir(gguf_dir):
            gguf_files = [f for f in os.listdir(gguf_dir) if f.endswith(".gguf")]
            if gguf_files:
                exports["formats"]["gguf"] = {"path": gguf_dir, "files": gguf_files}

    return exports


@router.get("/runs")
async def list_training_runs():
    """List ALL training runs (across all projects)."""
    from finetune_studio.db.runs import list_runs
    return list_runs()


@router.get("/runs/{pid}")
async def list_training_runs_for_project(pid: str):
    """List training runs for a specific project."""
    from finetune_studio.db.runs import list_runs
    return list_runs(pid)
