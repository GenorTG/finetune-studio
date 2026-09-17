"""Training loss must survive child → parent → DB → run list (LOSS column '—' bug).

Root causes covered:
- the spawn worker's state payload omitted ``final_loss``;
- the parent copied only a fixed key list that also omitted it;
- the Trainer's end-of-train summary log (no ``loss`` key) reset loss to 0;
- ``update_run`` silently dropped ``metrics=``;
- ``list_runs`` overwrote the ``final_loss`` column with a key that never exists.
"""

from __future__ import annotations

import time
from typing import Any

from finetune_studio.training.engine import (
    TrainingConfig,
    TrainingEngine,
    TrainingState,
    apply_trainer_log,
)
from finetune_studio.training.worker import _state_payload


def _wait_until(pred: Any, timeout: float = 5.0) -> None:
    deadline = time.time() + timeout
    while time.time() < deadline:
        if pred():
            return
        time.sleep(0.05)
    raise AssertionError(f"condition not met within {timeout}s")


def test_worker_payload_carries_final_loss() -> None:
    payload = _state_payload(TrainingState(status="done", loss=0.42, final_loss=0.42))
    assert payload["final_loss"] == 0.42


def test_parent_state_receives_final_loss_from_child() -> None:
    def fake_worker(_c: dict, _d: list, _p: str, out_queue: Any, _s: Any) -> None:
        out_queue.put({"op": "state", "state": {
            "status": "done", "current_step": 12, "total_steps": 12,
            "loss": 0.37, "final_loss": 0.37, "message": "done", "error": "",
            "log_lines": [],
        }})
        out_queue.put({"op": "done"})

    eng = TrainingEngine()
    finals: list[float | None] = []
    eng.on_update(lambda s: finals.append(s.final_loss))
    eng.start(TrainingConfig(model_path="m", output_dir="out", unsloth=False),
              [{"messages": []}], "", _worker_target=fake_worker)
    _wait_until(lambda: eng._listener is None)
    assert eng.state.final_loss == 0.37
    assert 0.37 in finals


def test_step_log_updates_loss_and_final_loss() -> None:
    state = TrainingState(status="training")
    apply_trainer_log(state, {"loss": 1.23456, "learning_rate": 1e-4},
                      global_step=5, epoch=0.5, total_steps=10, elapsed=10.0)
    assert state.loss == 1.2346
    assert state.final_loss == 1.2346
    assert state.current_step == 5
    assert state.eta == 10.0
    assert state.log_lines[-1].startswith("Step 5/10 | loss=1.2346")


def test_summary_and_eval_logs_do_not_reset_loss_to_zero() -> None:
    state = TrainingState(status="training")
    apply_trainer_log(state, {"loss": 0.8, "learning_rate": 2e-5},
                      global_step=9, epoch=2.9, total_steps=10, elapsed=9.0)
    # Trainer end-of-train summary: no "loss" key.
    apply_trainer_log(state, {"train_runtime": 12.0, "train_loss": 1.1},
                      global_step=10, epoch=3.0, total_steps=10, elapsed=12.0)
    assert state.loss == 0.8
    assert state.final_loss == 0.8
    assert state.learning_rate == 2e-5
    # Evaluation log: eval_loss is reported, training loss untouched.
    apply_trainer_log(state, {"eval_loss": 0.95},
                      global_step=10, epoch=3.0, total_steps=10, elapsed=12.5)
    assert state.loss == 0.8
    assert "eval_loss=0.95" in state.log_lines[-1]


def test_update_run_persists_metrics_and_list_runs_reports_final_loss(mock_settings) -> None:
    from finetune_studio import db

    pid = db.create_project(name="metrics")["id"]
    rid = db.create_run(pid, "Run")["id"]
    db.update_run(rid, status="done", final_loss=0.5123,
                  metrics={"total_steps": 12, "current_step": 12, "loss": 0.5123})

    run = db.get_run(rid)
    assert run["final_loss"] == 0.5123
    assert run["metrics"]["total_steps"] == 12

    listed = db.list_runs(pid)[0]
    assert listed["final_loss"] == 0.5123
    assert listed["metrics"]["current_step"] == 12


def test_list_runs_falls_back_to_metrics_loss_for_legacy_rows(mock_settings) -> None:
    from finetune_studio import db

    pid = db.create_project(name="legacy")["id"]
    rid = db.create_run(pid, "Run")["id"]
    db.update_run(rid, status="done", metrics={"loss": 0.77})
    assert db.list_runs(pid)[0]["final_loss"] == 0.77
