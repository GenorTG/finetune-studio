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

import asyncio
import json


async def training_events(engine):
    last_step = -1
    while engine.state.status in ("loading", "training", "saving"):
        if engine.state.current_step != last_step or engine.state.status != "training":
            last_step = engine.state.current_step
            data = json.dumps({
                "status": engine.state.status,
                "step": engine.state.current_step,
                "total_steps": engine.state.total_steps,
                "loss": engine.state.loss,
                "learning_rate": engine.state.learning_rate,
                "epoch": engine.state.epoch,
                "elapsed": engine.state.elapsed,
                "eta": engine.state.eta,
                "message": engine.state.message,
            })
            yield f"data: {data}\n\n"
        await asyncio.sleep(0.5)
    data = json.dumps({
        "status": engine.state.status,
        "step": engine.state.current_step,
        "total_steps": engine.state.total_steps,
        "loss": engine.state.loss,
        "message": engine.state.message,
        "error": engine.state.error,
        "log_lines": engine.state.log_lines[-20:],
    })
    yield f"data: {data}\n\n"
