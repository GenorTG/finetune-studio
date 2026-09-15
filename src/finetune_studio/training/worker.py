"""Training child process entry (E2E-40).

Runs Unsloth / heavy training OUTSIDE the uvicorn process so monkey-patches
never poison in-server inference. Parent talks over a multiprocessing Queue.

Why spawn (not fork, not ``python -m`` subprocess):
- ``spawn`` gives a clean interpreter with no inherited CUDA/thread state and
  no accidental ``unsloth`` import from the parent.
- Same venv/sys.path as the parent without re-resolving a module CLI.
- ``Process.terminate()`` makes Stop reliable during loading/saving (E2E-28).
"""

from __future__ import annotations

import logging
import os
from dataclasses import asdict, fields
from typing import Any

log = logging.getLogger(__name__)

# Apply before any datasets/unsloth import in this process (E2E-27).
os.environ.setdefault("UNSLOTH_DATASET_NUM_PROC", "0")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")


def _config_from_dict(raw: dict[str, Any]) -> Any:
    from finetune_studio.training.engine import TrainingConfig

    allowed = {f.name for f in fields(TrainingConfig)}
    return TrainingConfig(**{k: v for k, v in raw.items() if k in allowed})


def _state_payload(state: Any) -> dict[str, Any]:
    return {
        "status": state.status,
        "current_step": state.current_step,
        "total_steps": state.total_steps,
        "loss": state.loss,
        "learning_rate": state.learning_rate,
        "epoch": state.epoch,
        "elapsed": state.elapsed,
        "eta": state.eta,
        "message": state.message,
        "error": state.error,
        "log_lines": list(state.log_lines),
    }


def training_worker(
    config_dict: dict[str, Any],
    training_data: list,
    system_prompt: str,
    out_queue: Any,
    stop_event: Any,
) -> None:
    """Child entry: run ``TrainingEngine._train`` and push state to ``out_queue``.

    Messages:
      ``{"op": "state", "state": {...}}`` — progress / status snapshot
      ``{"op": "done"}`` — child finished (check last state for error/stopped)
    """
    os.environ.setdefault("UNSLOTH_DATASET_NUM_PROC", "0")
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

    from finetune_studio.training.engine import TrainingEngine, _format_exc

    eng = TrainingEngine()
    eng.config = _config_from_dict(config_dict)
    # Parent owns DB persistence via on_update; keep current_run_id unset here.
    eng.current_run_id = None

    class _StopBridge:
        """threading.Event-compatible wrapper over a multiprocessing Event."""

        def is_set(self) -> bool:
            return bool(stop_event.is_set())

        def set(self) -> None:
            stop_event.set()

        def clear(self) -> None:
            try:
                stop_event.clear()
            except Exception:  # noqa: BLE001, S110
                pass

    eng._stop_event = _StopBridge()  # type: ignore[assignment]

    def _push(state: Any) -> None:
        try:
            out_queue.put({"op": "state", "state": _state_payload(state)})
        except Exception:  # noqa: BLE001
            log.exception("Failed to push training state to parent")

    eng.on_update(_push)

    try:
        eng._train(training_data, system_prompt)
    except Exception as exc:  # noqa: BLE001
        # _train normally catches; this is belt-and-suspenders for empty msgs.
        msg = _format_exc(exc)
        eng.state.status = "error"
        eng.state.error = msg
        eng.state.message = msg
        _push(eng.state)
        log.exception("Training worker crashed: %s", msg)
    finally:
        try:
            out_queue.put({"op": "done"})
        except Exception:  # noqa: BLE001, S110
            pass


def config_to_dict(config: Any) -> dict[str, Any]:
    """Serialize ``TrainingConfig`` for the spawn queue."""
    return asdict(config)
