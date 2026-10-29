"""A run whose loss or gradient becomes NaN/inf stops and fails in plain words instead of finishing "done".

Found by the Quillmoor benchmark (2026-10-29): Qwen3.5-0.8B, SFT LoRA, healthy to step 810 (loss 0.32), then every
gradient NaN from step 820; the trainer logged loss 0.0 for the last 150 steps, the run ended "done" and the merged
model and its GGUF were built from the broken weights.
"""
from __future__ import annotations

import inspect
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from finetune_studio.training import engine as eng_mod
from finetune_studio.training.engine import TrainingConfig, TrainingDiverged, TrainingEngine, _stop_training_callback


class Control:
    should_training_stop = False


def _log(eng: TrainingEngine, logs: dict, step: int = 820, out: str = "/nonexistent") -> Control:
    control = Control()
    state = MagicMock(global_step=step)
    args = MagicMock(output_dir=out)
    _stop_training_callback(eng).on_log(args, state, control, logs=logs)
    return control


@pytest.mark.parametrize("logs", [{"loss": 0.69, "grad_norm": float("nan")}, {"loss": float("nan")},
                                  {"loss": float("inf"), "grad_norm": 1.0}])
def test_a_non_finite_log_stops_the_run(logs: dict) -> None:
    eng = TrainingEngine()
    control = _log(eng, logs)
    assert control.should_training_stop is True
    assert eng._diverged and eng._diverged["step"] == 820


def test_finite_logs_and_logs_without_numbers_change_nothing() -> None:
    eng = TrainingEngine()
    for logs in ({"loss": 0.316, "grad_norm": 0.7}, {"eval_loss": 0.4}, {"loss": 0.0, "grad_norm": 0.0}, {}):
        assert _log(eng, logs).should_training_stop is False
    assert eng._diverged is None


def test_the_step_after_a_divergence_stops_too() -> None:
    eng = TrainingEngine()
    _log(eng, {"grad_norm": float("nan")})
    control = Control()
    _stop_training_callback(eng).on_step_end(MagicMock(), MagicMock(), control)
    assert control.should_training_stop is True


def test_message_names_the_step_and_the_last_good_checkpoint(tmp_path: Path) -> None:
    for n in (700, 800, 900):
        (tmp_path / f"checkpoint-{n}").mkdir()
    msg = eng_mod._divergence_message({"step": 820, "value": "grad_norm", "output_dir": str(tmp_path)})
    assert "step 820" in msg and "gradient became NaN" in msg and "checkpoint-800" in msg
    assert "nothing was saved, merged or exported" in msg
    none = eng_mod._divergence_message({"step": 50, "value": "loss", "output_dir": str(tmp_path)})
    assert "loss became NaN" in none and "No checkpoint was saved before it" in none


def test_every_training_path_checks_right_after_train() -> None:
    for path in ("_train_standard", "_train_unsloth", "_train_preference"):
        src = inspect.getsource(getattr(TrainingEngine, path))
        after = src.split("trainer.train()", 1)[1]
        assert after.lstrip().startswith("if self._diverged:"), path


def test_a_diverged_run_ends_in_error_and_saves_nothing(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    eng = TrainingEngine()
    eng.config = TrainingConfig(model_path="m", output_dir=str(tmp_path), unsloth=False, merge_on_save=True)
    saved: list[str] = []
    monkeypatch.setattr(eng, "_persist_run_output", lambda: saved.append("output"))
    monkeypatch.setattr(eng, "_persist_run_error", lambda msg: saved.append(f"error: {msg}"))
    msg = eng_mod._divergence_message({"step": 820, "value": "grad_norm", "output_dir": str(tmp_path)})

    def train_standard(train_data, val_data) -> None:
        raise TrainingDiverged(msg)

    monkeypatch.setattr(eng, "_train_standard", train_standard)
    rows = [{"messages": [{"role": "user", "content": f"q{i}"}, {"role": "assistant", "content": f"a{i}"}]} for i in range(5)]
    eng._train(rows, "")
    assert eng.state.status == "error"
    assert eng.state.error == msg                     # plain words, no exception class name
    assert saved == [f"error: {msg}"]                 # the run row gets the reason; no output is recorded
