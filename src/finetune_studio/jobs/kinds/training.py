"""The training job: runs ``TrainingEngine._train`` in the job process (owned by the supervisor, not the web app).

Params: ``config`` (``training.worker.config_to_dict``), ``system_prompt``, ``run_id`` (the ``training_runs`` row),
``output_dir``, ``project_id``. The rows to train on are in ``data.jsonl`` next to the spec (too big for params).
The job persists the run row itself (``run_persistence``), so progress is saved even while no web process exists;
the live state for the page goes to ``state.json`` (read by ``training.remote.RemoteTrainingEngine``).
"""
from __future__ import annotations

import json
import logging
from typing import Any

from finetune_studio.jobs.runner import JobContext

log = logging.getLogger(__name__)
LIVE_LOG_LINES = 200


def state_payload(state: Any) -> dict[str, Any]:
    """Everything the training page shows, JSON-safe."""
    return {
        "status": state.status, "current_step": state.current_step, "total_steps": state.total_steps,
        "loss": state.loss, "final_loss": state.final_loss,
        "eval_loss": getattr(state, "eval_loss", None), "best_eval_loss": getattr(state, "best_eval_loss", None),
        "learning_rate": state.learning_rate, "epoch": state.epoch, "elapsed": state.elapsed, "eta": state.eta,
        "message": state.message, "error": state.error,
        "log_lines": list(state.log_lines[-LIVE_LOG_LINES:]),
        "pref_metrics": dict(getattr(state, "pref_metrics", None) or {}),
    }


def write_rows(ctx_dir, rows: list) -> None:
    with (ctx_dir / "data.jsonl").open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def _read_rows(ctx: JobContext) -> list:
    with ctx.path("data.jsonl").open(encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def run(ctx: JobContext) -> dict[str, Any]:
    import os
    os.environ.setdefault("UNSLOTH_DATASET_NUM_PROC", "0")
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    from finetune_studio.training.accel_plan import apply_alloc_conf
    apply_alloc_conf()

    from finetune_studio.training.engine import TrainingEngine, _format_exc
    from finetune_studio.training.run_persistence import make_run_state_persister
    from finetune_studio.training.worker import _config_from_dict

    p = ctx.params
    rows = _read_rows(ctx)
    eng = TrainingEngine()
    eng.config = _config_from_dict(p["config"])
    project_id = p.get("project_id")
    eng.current_run_id = f"{project_id}-{p['run_id']}" if project_id else p["run_id"]
    eng.current_project_id = project_id
    eng.current_db_run_id = p["run_id"]
    eng._stop_event = ctx.stop                          # SIGTERM (Stop button) -> trainer stops cooperatively
    eng.on_update(make_run_state_persister(p["run_id"], p["output_dir"]))

    last_status = {"v": None}

    def _publish(state: Any) -> None:
        changed = state.status != last_status["v"]
        last_status["v"] = state.status
        ctx.publish({**state_payload(state), "run_id": p["run_id"], "project_id": project_id}, force=changed)

    eng.on_update(_publish)
    eng.state.status, eng.state.message = "loading", "Loading the base model..."
    eng._notify()
    try:
        eng._train(rows, p.get("system_prompt", ""))
    except Exception as exc:                           # _train catches its own errors; this is the backstop
        msg = _format_exc(exc)
        log.exception("training crashed: %s", msg)
        eng.state.status, eng.state.error, eng.state.message = "error", msg, msg
        eng._notify()
    _publish(eng.state)
    ctx.publish({**state_payload(eng.state), "run_id": p["run_id"], "project_id": project_id}, force=True)
    status = {"done": "done", "stopped": "stopped"}.get(eng.state.status, "failed")
    return {"status": status, "run_id": p["run_id"], "final_loss": eng.state.final_loss,
            **({"error": eng.state.error} if status == "failed" else {})}
