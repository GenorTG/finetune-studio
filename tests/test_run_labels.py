"""Runs and trained exports must be distinguishable in every selector.

Every run on the same dataset used to be named ``Run · <dataset file>`` and
every trained GGUF was listed by its bare filename, so Testing / Chat /
Inference / Export showed identical entries for different runs.
"""

from __future__ import annotations

from pathlib import Path

from finetune_studio.models.registry import scan_models


def test_init_db_tags_legacy_auto_run_names_with_run_id(mock_settings) -> None:
    from finetune_studio import db

    pid = db.create_project(name="labels")["id"]
    auto = db.create_run(pid, "Run · data.jsonl")["id"]
    custom = db.create_run(pid, "My tuned run")["id"]

    db.init_db()
    db.init_db()  # idempotent

    assert db.get_run(auto)["name"] == f"Run {auto} · data.jsonl"
    assert db.get_run(custom)["name"] == "My tuned run"


def test_trained_gguf_exports_are_labelled_with_their_run_dir(tmp_path: Path) -> None:
    for run in ("8587cee6", "quality-v2"):
        gguf_dir = tmp_path / "output" / run / "gguf"
        gguf_dir.mkdir(parents=True)
        (gguf_dir / "model-q8_0.gguf").write_bytes(b"GGUF" + b"\0" * 64)

    names = sorted(m.name for m in scan_models([str(tmp_path / "output")]))

    assert names == ["8587cee6/model-q8_0.gguf", "quality-v2/model-q8_0.gguf"]
