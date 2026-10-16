"""Training progress monitoring.

WHAT THIS FILE DOES
==================
Tracks training metrics and reports them:
  - Loss (lower = better)
  - Learning rate over time
  - Tokens per second (training speed)
  - GPU memory usage
  - ETA (estimated time remaining)
  - Loss curves (saved as plots)

KEY CONCEPTS
============
- Loss: the error the model makes on training data. Should decrease
  over time. If it plateaus too early, you might be overfitting.
- Learning rate schedule: often we decrease the learning rate as
  training progresses (warmup, then decay).
- GPU memory: training large models can use 20-24GB. Running out
  = crash. We monitor to warn before crash.
"""

from __future__ import annotations

import asyncio
from typing import Any

from finetune_studio.webui.live_sse import sse_comment, sse_data

ACTIVE_STATUSES = ("loading", "training", "saving", "running")


def _idle_snapshot(project_id: str | None) -> dict[str, Any]:
    return {
        "status": "idle", "step": 0, "current_step": 0, "total_steps": 0, "loss": 0.0, "final_loss": None,
        "eval_loss": None, "best_eval_loss": None, "learning_rate": 0.0, "epoch": 0.0, "elapsed": 0.0, "eta": 0.0,
        "message": "", "error": "", "project_id": project_id, "log_lines": [], "pref_metrics": {}, "running": False,
        "other_project_running": False,
    }


def training_snapshot(engine: Any, project_id: str | None = None) -> dict[str, Any]:
    """Build a JSON-serializable status dict from the training engine.

    The engine is process-wide (one run at a time, and the last run's final state stays readable). With
    ``project_id`` the snapshot is scoped to that project: another project's run, live or finished, is reported as
    ``idle`` here (``other_project_running`` says when the card is busy with it), so a project page never shows a
    neighbour's step log as its own.
    """
    s = engine.state
    owner = getattr(engine, "current_project_id", None)
    if project_id is not None and owner and owner != project_id:
        out = _idle_snapshot(project_id)
        out["other_project_running"] = s.status in ACTIVE_STATUSES
        return out
    return {
        "status": s.status,
        "step": s.current_step,
        "current_step": s.current_step,
        "total_steps": s.total_steps,
        "loss": s.loss,
        "final_loss": s.final_loss,
        "eval_loss": getattr(s, "eval_loss", None),
        "best_eval_loss": getattr(s, "best_eval_loss", None),
        "learning_rate": s.learning_rate,
        "epoch": s.epoch,
        "elapsed": s.elapsed,
        "eta": s.eta,
        "message": s.message,
        "error": s.error,
        "project_id": owner,
        "log_lines": list(s.log_lines[-40:]),
        "pref_metrics": dict(getattr(s, "pref_metrics", None) or {}),
        "running": s.status in ACTIVE_STATUSES,
        "other_project_running": False,
    }


async def training_events(engine: Any, project_id: str | None = None):
    """SSE generator: emit status whenever training progress changes.

    Keeps the connection open with keepalive comments so the client does
    not need a 2-second full redraw poll. Clients close on navigation.
    ``project_id`` scopes the snapshots like ``training_snapshot``.
    """
    last_key: tuple[Any, ...] | None = None
    while True:
        payload = training_snapshot(engine, project_id)
        key = (
            payload["status"],
            payload["step"],
            payload.get("loss"),
            payload.get("message"),
            payload.get("error"),
            len(payload.get("log_lines") or []),
            (payload.get("log_lines") or [None])[-1],
        )
        if key != last_key:
            last_key = key
            yield sse_data(payload)
        else:
            yield sse_comment()
        await asyncio.sleep(0.5)
