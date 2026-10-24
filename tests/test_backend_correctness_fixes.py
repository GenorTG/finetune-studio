"""Regression tests for the E2E backend correctness findings (HANDOFF item 7)."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from finetune_studio import db
from finetune_studio.config import settings
from finetune_studio.data.fs import paths
from finetune_studio.webui.app import app


@pytest.fixture
def env(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    monkeypatch.setattr(settings, "db_path", str(data_dir / "x.db"))
    monkeypatch.setattr(settings, "data_dir", str(data_dir))
    root = tmp_path / "fts_root"
    (root / "projects").mkdir(parents=True)
    monkeypatch.setattr(paths, "_ROOT", root)
    monkeypatch.setattr(paths, "_PROJECTS", root / "projects")
    cwd = tmp_path / "cwd"
    cwd.mkdir()
    monkeypatch.chdir(cwd)
    db.init_db()
    client = TestClient(app)
    pid = client.post("/api/projects", json={"name": "bc", "base_model": "x/t"}).json()["id"]
    return client, pid, paths.project_dir(pid), cwd


def test_relative_path_and_output_both_resolve_in_project(env) -> None:
    client, pid, pdir, cwd = env
    pdir.mkdir(parents=True, exist_ok=True)
    row = json.dumps({"question": "q", "answer": "a"}) + "\n"
    (pdir / "in.jsonl").write_text(row, encoding="utf-8")
    (cwd / "in.jsonl").write_text(row * 2, encoding="utf-8")  # decoy in cwd
    r = client.post("/api/data/convert", json={
        "path": "in.jsonl", "output": "out.json", "target_format": "json", "project_id": pid})
    assert r.status_code == 200, r.text
    body = r.json()
    assert Path(body["path"]) == (pdir / "in.jsonl").resolve()
    assert len(json.loads((pdir / "out.json").read_text(encoding="utf-8"))) == 1
    assert not (cwd / "out.json").exists()


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
