"""Training tab — start/stop training, monitor progress."""

from fastapi import APIRouter, Request
from fastapi.responses import PlainTextResponse, StreamingResponse

from finetune_studio.training.data import load_jsonl
from finetune_studio.training.engine import TrainingConfig
from finetune_studio.training.monitor import training_events
from finetune_studio.webui.app import training_engine

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

@router.post("/start")
async def start_training(request: Request):
    body = await request.json()
    config = TrainingConfig(
        model_path=body.get("model_path", ""),
        output_dir=body.get("output_dir", "output"),
        lora_rank=int(body.get("lora_rank", 64)),
        learning_rate=float(body.get("learning_rate", 8e-5)),
        num_epochs=int(body.get("num_epochs", 4)),
        batch_size=int(body.get("batch_size", 2)),
        max_seq_length=int(body.get("max_seq_length", 2048)),
    )
    data_path = body.get("data_path", "")
    if not data_path:
        return {"error": "No data_path provided"}
    if not config.model_path:
        return {"error": "No model_path provided"}
    training_data = load_jsonl(data_path)
    system_prompt = body.get("system_prompt", "")
    training_engine.start(config, training_data, system_prompt)
    return {"status": "started", "steps": training_engine.state.total_steps}

@router.post("/stop")
async def stop_training():
    training_engine.stop()
    return {"status": "stopping"}
