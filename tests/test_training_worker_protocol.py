"""E2E-40: training worker protocol — parent state, stop, crash formatting."""

from __future__ import annotations

import time
from typing import Any

from finetune_studio.training.engine import TrainingConfig, TrainingEngine


def _wait_until(pred, timeout: float = 5.0, interval: float = 0.05) -> None:
    deadline = time.time() + timeout
    while time.time() < deadline:
        if pred():
            return
        time.sleep(interval)
    raise AssertionError(f"condition not met within {timeout}s")


def test_progress_messages_update_parent_state() -> None:
    def fake_worker(
        _config: dict[str, Any],
        _data: list,
        _prompt: str,
        out_queue: Any,
        _stop_event: Any,
    ) -> None:
        out_queue.put(
            {
                "op": "state",
                "state": {
                    "status": "training",
                    "current_step": 3,
                    "total_steps": 10,
                    "loss": 1.25,
                    "learning_rate": 1e-4,
                    "epoch": 0.5,
                    "elapsed": 1.0,
                    "eta": 2.0,
                    "message": "Step 3",
                    "error": "",
                    "log_lines": ["Step 3/10 | loss=1.25"],
                },
            }
        )
        out_queue.put({"op": "done"})

    eng = TrainingEngine()
    seen: list[str] = []
    eng.on_update(lambda s: seen.append(s.status))
    eng.start(
        TrainingConfig(model_path="m", output_dir="out", unsloth=False),
        [{"messages": []}],
        "",
        _worker_target=fake_worker,
    )
    _wait_until(lambda: eng.state.status == "training" and eng.state.current_step == 3)
    _wait_until(lambda: eng._listener is None, timeout=5.0)
    assert eng.state.loss == 1.25
    assert eng.state.log_lines == ["Step 3/10 | loss=1.25"]
    assert "training" in seen


def test_stop_terminates_child_and_marks_stopped() -> None:
    def sticky_worker(
        _config: dict[str, Any],
        _data: list,
        _prompt: str,
        out_queue: Any,
        stop_event: Any,
    ) -> None:
        out_queue.put(
            {
                "op": "state",
                "state": {
                    "status": "loading",
                    "message": "Loading model…",
                    "current_step": 0,
                    "total_steps": 0,
                    "loss": 0.0,
                    "learning_rate": 0.0,
                    "epoch": 0.0,
                    "elapsed": 0.0,
                    "eta": 0.0,
                    "error": "",
                    "log_lines": [],
                },
            }
        )
        # Hang until parent signals stop (loading-phase Stop case).
        while not stop_event.is_set():
            time.sleep(0.05)
        # Do not emit a terminal state — parent stop()/listener owns that.

    eng = TrainingEngine()
    eng.start(
        TrainingConfig(model_path="m", output_dir="out", unsloth=False),
        [],
        "",
        _worker_target=sticky_worker,
    )
    _wait_until(lambda: eng.state.status == "loading")
    proc = eng._process
    assert proc is not None and proc.is_alive()  # type: ignore[union-attr]
    eng.stop()
    _wait_until(lambda: eng.state.status == "stopped")
    assert eng.state.message == "Stopped by user"
    _wait_until(lambda: not proc.is_alive(), timeout=5.0)  # type: ignore[union-attr]


def test_child_crash_empty_exception_sets_typed_error() -> None:
    def crashing_worker(
        _config: dict[str, Any],
        _data: list,
        _prompt: str,
        out_queue: Any,
        _stop_event: Any,
    ) -> None:
        out_queue.put(
            {
                "op": "state",
                "state": {
                    "status": "error",
                    "error": "NotImplementedError",
                    "message": "NotImplementedError",
                    "current_step": 0,
                    "total_steps": 0,
                    "loss": 0.0,
                    "learning_rate": 0.0,
                    "epoch": 0.0,
                    "elapsed": 0.0,
                    "eta": 0.0,
                    "log_lines": [],
                },
            }
        )
        out_queue.put({"op": "done"})

    eng = TrainingEngine()
    eng.start(
        TrainingConfig(model_path="m", output_dir="out", unsloth=False),
        [],
        "",
        _worker_target=crashing_worker,
    )
    _wait_until(lambda: eng.state.status == "error")
    assert eng.state.error == "NotImplementedError"
    _wait_until(lambda: eng._listener is None, timeout=5.0)
