"""Persist one training run's state transitions to its ``training_runs`` row.

The WebUI owns a single process-wide ``TrainingEngine``. Every run must bind
exactly one DB callback: earlier callbacks are detached when a new run starts,
and a callback goes inert once its run reaches a terminal status, so a later
run can never rewrite an earlier run's status, finish time, or loss.
"""

from __future__ import annotations

import logging
import time
from collections.abc import Callable
from typing import Any

log = logging.getLogger(__name__)

ACTIVE_STATUSES: tuple[str, ...] = ("running", "training", "loading", "saving")
TERMINAL_STATUSES: tuple[str, ...] = ("done", "error", "stopped")


def make_run_state_persister(
    run_id: str,
    output_dir: str,
    *,
    update_run: Callable[..., Any] | None = None,
    clock: Callable[[], float] = time.time,
) -> Callable[[Any], None]:
    """Return an engine ``on_update`` callback bound to one run row."""
    started_at = clock()
    started_logged = False
    finished = False

    def _persist(state: Any) -> None:
        nonlocal started_logged, finished
        if finished:
            return
        status = getattr(state, "status", "")
        update: dict[str, Any] = {}
        if status in ACTIVE_STATUSES:
            update["status"] = status
            if not started_logged:
                started_logged = True
                update["started_at"] = started_at
        elif status in TERMINAL_STATUSES:
            finished = True
            now = clock()
            update["finished_at"] = now
            update["metrics"] = {
                "total_steps": state.total_steps,
                "current_step": state.current_step,
                "loss": state.loss,
                "final_loss": state.final_loss,
                "epoch": state.epoch,
                "elapsed": state.elapsed,
                "duration": max(0.0, now - started_at),
            }
            if state.final_loss is not None:
                update["final_loss"] = state.final_loss
            if status == "done":
                update["status"] = "done"
                update["output_path"] = output_dir
                if state.error:
                    # Soft post-train failure (merge / GGUF): keep done, surface it.
                    update["error"] = state.error[:2000]
                    update["notes"] = state.error[:2000]
            elif status == "stopped":
                update["status"] = "stopped"
                update["notes"] = "Stopped by user"
            else:
                update["status"] = "failed"
                update["error"] = (state.error or state.message or "training failed")[:2000]
        if not update:
            return
        writer = update_run
        if writer is None:
            from finetune_studio import db

            writer = db.update_run
        try:
            writer(run_id, **update)
        except Exception:
            # The engine callback must never die on a DB hiccup.
            log.exception("update_run failed for training run %s", run_id)

    return _persist


def attach_run(
    engine: Any,
    run_id: str,
    output_dir: str,
    *,
    project_id: str | None,
    update_run: Callable[..., Any] | None = None,
) -> None:
    """Tag ``engine`` with ``run_id`` and bind exactly one persister to it."""
    previous = getattr(engine, "_run_persister_unsubscribe", None)
    if callable(previous):
        previous()
    engine.current_run_id = f"{project_id}-{run_id}" if project_id else run_id
    engine.current_project_id = project_id
    engine.current_db_run_id = run_id
    engine._run_persister_unsubscribe = engine.on_update(
        make_run_state_persister(run_id, output_dir, update_run=update_run)
    )
