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


def training_snapshot(engine: Any) -> dict[str, Any]:
    """Build a JSON-serializable status dict from the training engine."""
    s = engine.state
    return {
        "status": s.status,
        "step": s.current_step,
        "current_step": s.current_step,
        "total_steps": s.total_steps,
        "loss": s.loss,
        "learning_rate": s.learning_rate,
        "epoch": s.epoch,
        "elapsed": s.elapsed,
        "eta": s.eta,
        "message": s.message,
        "error": s.error,
        "log_lines": list(s.log_lines[-40:]),
    }


async def training_events(engine: Any):
    """SSE generator: emit status whenever training progress changes.

    Keeps the connection open with keepalive comments so the client does
    not need a 2-second full redraw poll. Clients close on navigation.
    """
    last_key: tuple[Any, ...] | None = None
    while True:
        payload = training_snapshot(engine)
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
