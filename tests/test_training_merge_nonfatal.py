"""E2E-32: merge failure is non-fatal; empty NotImplementedError surfaces as type name."""
from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock

import pytest

from finetune_studio.training.engine import TrainingConfig, TrainingEngine, _format_exc


def test_format_exc_empty_notimplemented() -> None:
    assert _format_exc(NotImplementedError()) == "NotImplementedError"
    assert _format_exc(NotImplementedError("boom")) == "NotImplementedError: boom"


def test_do_merge_raising_marks_run_done_with_output_and_message(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    eng = TrainingEngine()
    out = tmp_path / "run-out"
    adapter = out / "adapter"
    adapter.mkdir(parents=True)
    (adapter / "adapter_config.json").write_text("{}", encoding="utf-8")

    eng.config = TrainingConfig(model_path="unsloth/qwen3-0.6b-unsloth-bnb-4bit", output_dir=str(out))
    eng.current_run_id = "testrun01"
    eng.state.status = "saving"

    updates: list[dict] = []

    def fake_update(rid: str, **fields: object) -> dict:
        updates.append({"id": rid, **fields})
        return {"id": rid, **fields}

    monkeypatch.setattr("finetune_studio.db.runs.update_run", fake_update)

    def boom(*_a: object, **_k: object) -> dict:
        raise NotImplementedError()

    monkeypatch.setattr(eng, "_do_merge", boom)

    # Simulate the post-train merge path used by _train_unsloth / _train_standard.
    eng._maybe_merge(MagicMock(), MagicMock(), str(out))
    eng.state.status = "done"
    if not (eng.state.message or "").startswith("Training complete —"):
        eng.state.message = "Training complete!"
    eng._persist_run_output()

    assert eng.state.status == "done"
    assert "merge failed" in eng.state.message.lower()
    assert "NotImplementedError" in eng.state.message
    assert updates, "expected DB persist"
    last = updates[-1]
    assert last["status"] == "done"
    assert last["output_path"] == str(out)
    assert "NotImplementedError" in (last.get("error") or last.get("notes") or "")


def test_train_exception_empty_notimplemented_sets_typed_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    eng = TrainingEngine()
    eng.config = TrainingConfig(output_dir="output/x", unsloth=False)

    def boom(_data: list) -> None:
        raise NotImplementedError()

    monkeypatch.setattr(eng, "_train_standard", boom)
    monkeypatch.setattr(
        "finetune_studio.training.data.format_for_sft",
        lambda data, _sp: data or [{"messages": []}],
    )
    monkeypatch.setattr(
        "finetune_studio.training.data.split_data",
        lambda data: (data, []),
    )

    eng._train([{"messages": [{"role": "user", "content": "q"}, {"role": "assistant", "content": "a"}]}], "")

    assert eng.state.status == "error"
    assert eng.state.error == "NotImplementedError"
    assert eng.state.message == "NotImplementedError"
    assert not eng.state.message.startswith("Error: ")
