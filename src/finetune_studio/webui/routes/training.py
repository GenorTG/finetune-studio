"""Training tab — start/stop training, monitor progress."""

import json
import os
from pathlib import Path

from fastapi import APIRouter, Request
from fastapi.responses import JSONResponse, PlainTextResponse

from finetune_studio.training.data import load_jsonl
from finetune_studio.training.engine import TrainingConfig
from finetune_studio.training.monitor import training_events
from finetune_studio.webui.app import training_engine
from finetune_studio.webui.live_sse import sse_response

router = APIRouter()


def _coerce_bool(value: object) -> bool:
    """Parse JSON/FormData bool-ish values (``"1"``, ``"true"``, ``true``, …)."""
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    s = str(value).strip().lower()
    if s in ("1", "true", "yes", "on"):
        return True
    if s in ("0", "false", "no", "off", ""):
        return False
    return bool(value)


def _optional_body_bool(body: dict, key: str, overrides: dict | None = None) -> bool | None:
    """Return coerced bool when ``key`` is present on body or overrides; else None."""
    if key in body:
        return _coerce_bool(body[key])
    if overrides is not None and key in overrides:
        return _coerce_bool(overrides[key])
    return None


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
    return {
        "status": s.status,
        "step": s.current_step,
        "total_steps": s.total_steps,
        "loss": s.loss,
        "learning_rate": s.learning_rate,
        "epoch": s.epoch,
        "elapsed": s.elapsed,
        "eta": s.eta,
        "message": s.message,
        "error": s.error,
        "log_lines": list(s.log_lines[-30:]),
    }

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
    """SSE live training status (preferred over polling ``/status``)."""
    return sse_response(training_events(training_engine))


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

    # Extract data_path and project_id FIRST
    data_path = body.get("data_path", "")
    dataset_id = body.get("dataset_id", "")
    project_id = body.get("project_id", "")
    if dataset_id and not data_path:
        ds = db.get_dataset(dataset_id)
        if not ds or ds.get("project_id") != project_id:
            return {"error": f"dataset {dataset_id!r} not found in this project"}
        data_path = ds["data_path"]
    if not data_path:
        return {"error": "No data_path / dataset_id provided"}

    preset_id = body.get("preset_id")
    overrides = body.get("overrides") or {}
    # Project training form omits unchecked boxes; default false (not true).
    merge_flag = _optional_body_bool(body, "merge_on_save", overrides)
    unsloth_flag = _optional_body_bool(body, "unsloth", overrides)
    if preset_id:
        try:
            config = _apply_preset(preset_id, overrides)
        except ValueError as e:
            return {"error": str(e)}
        if not config.model_path and body.get("model_path"):
            config.model_path = body["model_path"]
        if merge_flag is not None:
            config.merge_on_save = merge_flag
        if unsloth_flag is not None:
            config.unsloth = unsloth_flag
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
            # Omitted → standard TRL (no Unsloth status wording) for stock Qwen3-4B flow.
            unsloth=False if unsloth_flag is None else unsloth_flag,
            merge_on_save=False if merge_flag is None else merge_flag,
            export_gguf=bool(body.get("export_gguf", False)),
            gguf_quants=body.get("gguf_quants", ["f16", "q8_0", "q4_k_m", "q5_k_m"]),
            data_path=data_path,
            project_id=project_id,
            abliterate=bool(body.get("abliterate", False)),
            abliteration_strength=float(body.get("abliteration_strength", 1.0)),
            export_gptq=bool(body.get("export_gptq", False)),
            gptq_bits=int(body.get("gptq_bits", 4)),
            gptq_group_size=int(body.get("gptq_group_size", 128)),
            export_imatrix=bool(body.get("export_imatrix", False)),
            imatrix_calibration=body.get("imatrix_calibration", ""),
        )
    merge_on_save = config.merge_on_save

    if not config.model_path:
        return {"error": "No model_path provided"}
    training_data = load_jsonl(data_path)
    system_prompt = body.get("system_prompt", "")
    system_prompt_mode = body.get("system_prompt_mode", "bake")
    # The project training form sends only the mode; bake/runtime without a
    # prompt used to silently train with none (E2E-26). Use the project's.
    if not system_prompt and system_prompt_mode != "none" and project_id:
        system_prompt = (db.get_project(project_id) or {}).get("system_prompt", "") or ""

    # Create a run record
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
            "system_prompt_mode": system_prompt_mode,
        },
        system_prompt=system_prompt,
        system_prompt_mode=system_prompt_mode,
    )
    run_id = run["id"]
    # A bare "output" default is shared by every project and run, so each new
    # run silently overwrote the previous run's adapter/merged model (E2E-25).
    if (config.output_dir or "output").rstrip("/") == "output":
        config.output_dir = (
            f"output/projects/{project_id}/runs/{run_id}" if project_id else f"output/runs/{run_id}"
        )
    # Persist immediately so Export / Training UI can locate the run even when
    # later status callbacks omit output_path or use a composite engine run id.
    db.update_run(run_id, output_path=config.output_dir)
    run_started_at = _time.time()
    training_engine.current_run_id = f"{project_id}-{run_id}" if project_id else run_id
    training_engine.current_project_id = project_id
    training_engine.current_db_run_id = run_id

    state_started_logged = {"value": False}
    def _on_state_change(state):
        update: dict = {}
        active_states = ("running", "training", "loading", "saving")
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
            update["metrics"] = {
                "total_steps": state.total_steps,
                "current_step": state.current_step,
                "loss": state.loss,
                "epoch": state.epoch,
                "elapsed": state.elapsed,
            }
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
        except Exception:  # noqa: BLE001, S110
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
    """Export a trained run to deployable formats (standalone, post-training).

    Body:
        format: gguf | gptq | abliterated | merged (default: gguf).
                Use gptq / gguf / merged / abliterated.
        quants: GGUF quant list (default: f16, q8_0, q4_k_m, q5_k_m)
        force: overwrite existing exports (default: false)
        base_model: optional 16-bit base path/id for merge-at-export when
                    the run only has an adapter (merge_on_save=false)
    """
    from finetune_studio import db
    from finetune_studio.training.run_export import export_trained_run

    body = await request.json()
    fmt = body.get("format", "gguf")
    quants = body.get("quants", ["f16", "q8_0", "q4_k_m", "q5_k_m"])
    force = bool(body.get("force", False))
    base_model = body.get("base_model")
    if base_model is not None:
        base_model = str(base_model).strip() or None

    run = db.get_run(run_id)
    if not run:
        return JSONResponse(
            {"ok": False, "status": "failed", "error": "run not found"},
            status_code=404,
        )

    result = export_trained_run(
        run,
        fmt=str(fmt),
        quants=list(quants) if isinstance(quants, list) else None,
        force=force,
        base_model=base_model,
    )
    if result.get("error") or result.get("ok") is False:
        return JSONResponse(result, status_code=400)
    return result

@router.get("/runs/{run_id}/auto-suites")
async def list_auto_suites(run_id: str):
    """List all auto-generated suites for a training run."""
    from finetune_studio.db.connection import cursor
    with cursor() as c:
        rows = c.execute("SELECT id, suite_name, suite_path, case_count, categories_json, created_at FROM auto_suites WHERE run_id = ? ORDER BY created_at DESC", (run_id,)).fetchall()
    return [{"id": r[0], "suite_name": r[1], "suite_path": r[2], "case_count": r[3], "categories": json.loads(r[4]) if r[4] else {}, "created_at": r[5]} for r in rows]


@router.post("/runs/{run_id}/auto-suites/generate")
async def trigger_auto_suite(run_id: str):
    """Trigger auto-generation of a benchmark suite from training data."""
    from finetune_studio import db
    run = db.get_run(run_id)
    if not run:
        return {"error": "run not found"}
    data_path = (run.get("data_path") or "").strip()
    if not data_path:
        return {"error": "run has no data_path"}
    output_path = (run.get("output_path") or "").strip()
    if not output_path:
        return {"error": "run has no output_path"}
    from finetune_studio.testing.generate_suite import generate_suite_from_training_data
    result = generate_suite_from_training_data(data_path, output_path)
    if result.get("error"):
        return result
    # Record in DB
    from time import time as _time

    from finetune_studio.db.connection import cursor, new_id
    suite_id = new_id()
    with cursor() as c:
        c.execute(
            "INSERT INTO auto_suites (id, run_id, project_id, suite_name, suite_path, case_count, categories_json, created_at) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
            (suite_id, run_id, run.get("project_id", ""),
             result.get("suite_name", "auto"), result.get("suite_path", ""),
             result.get("case_count", 0),
             json.dumps(result.get("categories", {})), _time()),
        )
    return {"ok": True, "suite_id": suite_id, **result}


@router.post("/runs/{run_id}/abliterate")
async def abliterate_run(run_id: str):
    """Abliterate (de-censor) a trained model."""
    from finetune_studio import db
    run = db.get_run(run_id)
    if not run:
        return {"error": "run not found"}
    output_path = (run.get("output_path") or "").strip()
    if not output_path:
        return {"error": "run has no output_path"}
    merged_dir = os.path.join(output_path, "merged")
    if not os.path.isdir(merged_dir) or not os.listdir(merged_dir):
        return {"error": "no merged model to abliterate"}
    abliterated_dir = os.path.join(output_path, "abliterated")
    from finetune_studio.training.abliteration import abliterate_model
    result = abliterate_model(
        model_path=merged_dir,
        output_dir=abliterated_dir,
        strength=float(run.get("abliteration_strength", 1.0)),
    )
    if result.get("error"):
        return result
    # Convert numpy arrays to lists for JSON serialization
    clean_result = {}
    for k, v in result.items():
        if hasattr(v, 'tolist'):
            clean_result[k] = v.tolist()
        elif isinstance(v, (list, tuple)):
            clean_result[k] = [float(x) if hasattr(x, 'item') else x for x in v]
        else:
            clean_result[k] = v
    from time import time as _time

    from finetune_studio.db.connection import cursor, new_id
    abl_id = new_id()
    with cursor() as c:
        c.execute(
            "INSERT INTO abliteration_runs (id, run_id, project_id, model_path, output_path, strength, magnitude, status, created_at) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (abl_id, run_id, run.get("project_id", ""),
             merged_dir, abliterated_dir,
             float(run.get("abliteration_strength", 1.0)),
             float(clean_result.get("refusal_magnitude", 0.0)),
             "done", _time()),
        )
    return {"ok": True, "abliteration_id": abl_id, **clean_result}


@router.get("/runs/{run_id}/abliteration")
async def get_abliteration(run_id: str):
    """Get abliteration status for a run."""
    from finetune_studio.db.connection import cursor
    with cursor() as c:
        rows = c.execute("SELECT id, model_path, output_path, strength, magnitude, status, created_at FROM abliteration_runs WHERE run_id = ? ORDER BY created_at DESC", (run_id,)).fetchall()
    return [{"id": r[0], "model_path": r[1], "output_path": r[2], "strength": r[3], "magnitude": r[4], "status": r[5], "created_at": r[6]} for r in rows]


@router.post("/runs/{run_id}/quantize")
async def quantize_run(run_id: str, request: Request):
    """Export a trained model using advanced quantization."""
    from finetune_studio import db
    body = await request.json()
    method = body.get("method", "gptq")
    run = db.get_run(run_id)
    if not run:
        return {"error": "run not found"}
    output_path = (run.get("output_path") or "").strip()
    if not output_path:
        return {"error": "run has no output_path"}
    merged_dir = os.path.join(output_path, "merged")
    if not os.path.isdir(merged_dir) or not os.listdir(merged_dir):
        return {"error": "no merged model to quantize"}
    if method == "gptq":
        output_dir = os.path.join(output_path, "gptq")
        from finetune_studio.training.advanced_quant import quantize_gptq
        result = quantize_gptq(
            model_path=merged_dir,
            output_dir=output_dir,
            bits=int(body.get("bits", 4)),
            group_size=int(body.get("group_size", 128)),
        )
    elif method == "imatrix":
        output_dir = os.path.join(output_path, "imatrix")
        from finetune_studio.training.advanced_quant import quantize_gguf_imatrix
        result = quantize_gguf_imatrix(
            model_path=merged_dir,
            output_dir=output_dir,
            imatrix_path=body.get("imatrix_path", ""),
            quants=body.get("quants", ["q4_k_m", "q5_k_m", "q8_0"]),
        )
    else:
        return {"error": f"unknown method: {method}"}
    if result.get("error"):
        return result
    from time import time as _time

    from finetune_studio.db.connection import cursor, new_id
    q_id = new_id()
    with cursor() as c:
        c.execute(
            "INSERT INTO quant_exports (id, run_id, project_id, model_path, output_path, method, bits, group_size, size_bytes, status, created_at) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (q_id, run_id, run.get("project_id", ""),
             merged_dir, output_dir,
             method,
             int(body.get("bits", 4)),
             int(body.get("group_size", 128)),
             result.get("size_bytes", 0),
             "done", _time()),
        )
    return {"ok": True, "quantize_id": q_id, **result}


@router.get("/runs/{run_id}/quant-exports")
async def list_quant_exports(run_id: str):
    """List all quantization exports for a run."""
    from finetune_studio.db.connection import cursor
    with cursor() as c:
        rows = c.execute("SELECT id, model_path, output_path, method, bits, group_size, size_bytes, status, created_at FROM quant_exports WHERE run_id = ? ORDER BY created_at DESC", (run_id,)).fetchall()
    return [{"id": r[0], "model_path": r[1], "output_path": r[2], "method": r[3], "bits": r[4], "group_size": r[5], "size_bytes": r[6], "status": r[7], "created_at": r[8]} for r in rows]


@router.post("/runs/{run_id}/set-output")
async def set_run_output(run_id: str, request: Request):
    """Update a run's output_path."""
    from finetune_studio import db
    body = await request.json()
    output_path = body.get("output_path", "").strip()
    if not output_path:
        return {"error": "no output_path"}
    db.update_run(run_id, output_path=output_path)
    return {"ok": True, "run_id": run_id, "output_path": output_path}


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
