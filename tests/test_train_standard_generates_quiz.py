"""Every training path must leave the project a quiz.

Found in the live walkthrough: a default run (the standard path, because FTS_UNSLOTH=auto skips Unsloth) finished
'done' with no auto-generated suite, so the Testing page offered nothing to run for the model just trained.
"""
from __future__ import annotations

import inspect
from types import SimpleNamespace

from finetune_studio.training import engine as eng_mod


def test_both_training_paths_generate_the_quiz() -> None:
    for path in ("_train_standard", "_train_unsloth"):
        assert "_auto_suite_non_fatal()" in inspect.getsource(getattr(eng_mod.TrainingEngine, path)), path


def test_quiz_failure_is_reported_but_never_fails_the_run() -> None:
    e = object.__new__(eng_mod.TrainingEngine)
    e.state = SimpleNamespace(message="Saved", error="")
    e._notify = lambda: None

    def boom() -> dict:
        raise RuntimeError("disk full")

    e._auto_generate_suite = boom
    e._auto_suite_non_fatal()          # must not raise
    assert "auto-suite failed" in e.state.message and "disk full" in e.state.message
