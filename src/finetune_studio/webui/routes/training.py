"""Training tab — start/stop training, monitor progress."""

from fastapi import APIRouter, Request
from fastapi.responses import PlainTextResponse, StreamingResponse

from finetune_studio.training.data import load_jsonl
from finetune_studio.training.engine import TrainingConfig
from finetune_studio.training.monitor import training_events
from finetune_studio.webui.app import training_engine
from pathlib import Path

router = APIRouter()

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

@router.post("/start")
async def start_training(request: Request):
    from finetune_studio import db
    body = await request.json()
    merge_on_save = bool(body.get("merge_on_save"))
    config = TrainingConfig(
        model_path=body.get("model_path", ""),
        output_dir=body.get("output_dir", "output"),
        lora_rank=int(body.get("lora_rank", 64)),
        learning_rate=float(body.get("learning_rate", 8e-5)),
        num_epochs=int(body.get("num_epochs", 4)),
        batch_size=int(body.get("batch_size", 2)),
        max_seq_length=int(body.get("max_seq_length", 2048)),
        merge_on_save=merge_on_save,
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
