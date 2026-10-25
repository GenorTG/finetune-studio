"""Regression tests for the E2E backend correctness findings (HANDOFF item 7)."""

from __future__ import annotations

from pathlib import Path

import pytest


def test_preset_start_honours_top_level_alpha_and_accum(client, tmp_path, monkeypatch) -> None:
    import finetune_studio.webui.routes.training as tr
    from tests import test_training_start_guard as g

    eng = g._FakeEngine()
    monkeypatch.setattr(tr, "training_engine", eng)
    monkeypatch.setattr(Path, "home", classmethod(lambda cls: tmp_path))
    pid = g._project(client)
    model = tmp_path / "m"
    model.mkdir()
    r = g._start(client, pid, g._jsonl(tmp_path), str(model), preset_id="standard",
                 lora_alpha="48", gradient_accumulation_steps="3", lora_rank="24")
    assert r.json().get("status") == "started", r.text
    cfg = eng.started["config"]
    assert (cfg.lora_alpha, cfg.gradient_accumulation_steps, cfg.lora_rank) == (48, 3, 24)


def test_validate_rejects_empty_jsonl_nonzero_exit(tmp_path, capsys) -> None:
    import types

    from finetune_studio.cli.commands.validate import cmd_validate
    from finetune_studio.data.validator import validate_file

    empty = tmp_path / "empty.jsonl"
    empty.write_text("\n\n", encoding="utf-8")
    rep = validate_file(empty)
    assert rep["valid"] is False and any("no rows" in e for e in rep["errors"])
    with pytest.raises(SystemExit) as ei:
        cmd_validate(types.SimpleNamespace(files=[str(empty)], file=str(empty), path=str(empty)))
    assert ei.value.code == 1


def test_import_relabels_old_project_sources(tmp_path) -> None:
    import pandas as pd

    from finetune_studio.smart_brain.io import (
        read_json,
        relabel_imported_sources,
        write_json,
    )

    old = "/srv/someone/old-project/docs/a.md"
    pd.DataFrame({"source": [old, "rel.txt"], "text": ["a", "b"]}).to_parquet(
        tmp_path / "chunks.parquet", index=False)
    write_json(tmp_path / "manifest.json",
               {"extra": {"documents_meta": [{"document_id": "d", "source": old}]}})
    relabel_imported_sources(tmp_path)
    relabel_imported_sources(tmp_path)  # idempotent
    df = pd.read_parquet(tmp_path / "chunks.parquet")
    assert list(df["source"]) == ["imported:a.md", "rel.txt"]
    assert read_json(tmp_path / "manifest.json")["extra"]["documents_meta"][0]["source"] == "imported:a.md"
