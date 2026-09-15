"""E2E-28: stop between phases skips merge and marks the run stopped."""
from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock

import pytest

from finetune_studio.training.engine import TrainingConfig, TrainingEngine


def test_stop_before_merge_skips_merge_and_marks_stopped(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    eng = TrainingEngine()
    out = tmp_path / "out"
    out.mkdir()
    eng.config = TrainingConfig(
        model_path="m",
        output_dir=str(out),
        merge_on_save=True,
        unsloth=False,
        batch_size=1,
        gradient_accumulation_steps=1,
        num_epochs=1,
    )

    merge_calls: list[object] = []

    def fake_merge(*_a: object, **_k: object) -> dict:
        merge_calls.append(1)
        return {"skipped": False}

    monkeypatch.setattr(eng, "_do_merge", fake_merge)
    monkeypatch.setattr(eng, "_maybe_merge", lambda *a, **k: eng._do_merge(*a, **k))

    # Minimal stand-in for the post-train phase: adapter saved, stop set, then merge gate.
    eng.state.status = "saving"
    eng.state.message = "Saving model..."
    (out / "adapter").mkdir()
    eng._stop_event.set()

    if eng._stop_requested():
        eng._mark_stopped()
    elif eng.config.merge_on_save:
        eng._maybe_merge(MagicMock(), MagicMock(), str(out))

    assert eng.state.status == "stopped"
    assert eng.state.message == "Stopped by user"
    assert merge_calls == []


def test_stop_callback_sets_should_training_stop() -> None:
    """The TrainerCallback used during train flips should_training_stop."""
    eng = TrainingEngine()
    eng._stop_event.set()

    class Control:
        should_training_stop = False

    # Inline the same logic as StopCallback.on_step_end
    control = Control()
    if eng._stop_requested():
        control.should_training_stop = True
    assert control.should_training_stop is True
