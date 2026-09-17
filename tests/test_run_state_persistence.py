"""Per-run training state persistence (callbacks leaked across runs).

The TrainingEngine is a process-wide singleton. Each started run used to append
a new DB callback that was never removed, so an earlier run's callback kept
rewriting its own row with a later run's status/finish time/loss.
"""

from __future__ import annotations

from typing import Any

from finetune_studio.training.engine import TrainingEngine, TrainingState
from finetune_studio.training.run_persistence import (
    attach_run,
    make_run_state_persister,
)


class _Recorder:
    def __init__(self) -> None:
        self.calls: list[tuple[str, dict[str, Any]]] = []

    def __call__(self, run_id: str, **fields: Any) -> None:
        self.calls.append((run_id, fields))


def _clock(values: list[float]):
    it = iter(values)
    return lambda: next(it)


def test_persister_records_start_progress_and_done() -> None:
    rec = _Recorder()
    persist = make_run_state_persister("A", "out/A", update_run=rec, clock=_clock([100.0, 160.0]))
    persist(TrainingState(status="loading"))
    persist(TrainingState(status="training", current_step=3, total_steps=6, loss=1.5))
    persist(TrainingState(status="done", current_step=6, total_steps=6, loss=0.4, final_loss=0.4))

    assert rec.calls[0] == ("A", {"status": "loading", "started_at": 100.0})
    assert rec.calls[1] == ("A", {"status": "training"})
    run_id, done = rec.calls[2]
    assert run_id == "A"
    assert done["status"] == "done"
    assert done["final_loss"] == 0.4
    assert done["output_path"] == "out/A"
    assert done["finished_at"] == 160.0
    assert done["metrics"]["final_loss"] == 0.4
    assert done["metrics"]["current_step"] == 6
    assert done["metrics"]["duration"] == 60.0


def test_persister_ignores_states_after_terminal_status() -> None:
    rec = _Recorder()
    persist = make_run_state_persister("A", "out/A", update_run=rec, clock=_clock([1.0, 2.0]))
    persist(TrainingState(status="done", final_loss=0.3))
    persist(TrainingState(status="loading"))
    persist(TrainingState(status="training"))
    assert len(rec.calls) == 1


def test_error_is_persisted_as_failed_with_message() -> None:
    rec = _Recorder()
    persist = make_run_state_persister("A", "out/A", update_run=rec, clock=_clock([1.0, 2.0]))
    persist(TrainingState(status="error", error="CUDA out of memory"))
    fields = rec.calls[-1][1]
    assert fields["status"] == "failed"
    assert fields["error"] == "CUDA out of memory"


def test_stopped_is_persisted() -> None:
    rec = _Recorder()
    persist = make_run_state_persister("A", "out/A", update_run=rec, clock=_clock([1.0, 2.0]))
    persist(TrainingState(status="stopped", message="Stopped by user"))
    fields = rec.calls[-1][1]
    assert fields["status"] == "stopped"
    assert fields["notes"] == "Stopped by user"


def test_on_update_returns_unsubscribe() -> None:
    eng = TrainingEngine()
    seen: list[str] = []
    unsubscribe = eng.on_update(lambda s: seen.append(s.status))
    eng._notify()
    unsubscribe()
    eng._notify()
    assert seen == ["idle"]


def test_attach_run_replaces_previous_runs_persister() -> None:
    rec = _Recorder()
    eng = TrainingEngine()
    attach_run(eng, "A", "out/A", project_id="p", update_run=rec)
    attach_run(eng, "B", "out/B", project_id="p", update_run=rec)
    eng.state = TrainingState(status="training")
    eng._notify()
    assert {run_id for run_id, _ in rec.calls} == {"B"}
    assert eng.current_db_run_id == "B"
    assert eng.current_run_id == "p-B"
