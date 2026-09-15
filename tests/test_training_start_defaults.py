"""Regression E2E-25 / E2E-26 for POST /api/training/start.

E2E-25  output_dir defaulted to the shared "output" dir, so every run in every
        project overwrote the previous run's adapter + merged model.
E2E-26  the project training form sends system_prompt_mode=bake but no prompt,
        so the project's system prompt was never baked in.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest


class _FakeEngine:
    def __init__(self) -> None:
        self.started: dict = {}
        self.current_run_id = ""
        self.current_project_id = ""
        self.current_db_run_id = ""

        class _State:
            total_steps = 0

        self.state = _State()

    def on_update(self, cb: object) -> None:
        self.cb = cb

    def start(self, config: object, data: list, system_prompt: str) -> None:
        self.started = {"config": config, "data": data, "system_prompt": system_prompt}


@pytest.fixture
def fake_engine(monkeypatch: pytest.MonkeyPatch) -> _FakeEngine:
    import finetune_studio.webui.routes.training as training_routes

    eng = _FakeEngine()
    monkeypatch.setattr(training_routes, "training_engine", eng)
    return eng


def _project(client, system_prompt: str) -> str:
    r = client.post("/api/projects", json={"name": "Start Defaults", "system_prompt": system_prompt})
    assert r.status_code in (200, 201), r.text
    return r.json()["id"]


def _jsonl(tmp_path: Path) -> str:
    p = tmp_path / "qa.jsonl"
    row = {"messages": [{"role": "user", "content": "q"}, {"role": "assistant", "content": "a"}]}
    p.write_text(json.dumps(row) + "\n", encoding="utf-8")
    return str(p)


def test_default_output_dir_is_scoped_per_run(client, fake_engine: _FakeEngine, tmp_path: Path) -> None:
    pid = _project(client, "")
    r = client.post("/api/training/start", json={
        "project_id": pid, "data_path": _jsonl(tmp_path), "model_path": "m", "output_dir": "output",
    })
    assert r.status_code == 200, r.text
    run_id = r.json()["run_id"]
    assert fake_engine.started["config"].output_dir == f"output/projects/{pid}/runs/{run_id}"


def test_explicit_output_dir_is_kept(client, fake_engine: _FakeEngine, tmp_path: Path) -> None:
    pid = _project(client, "")
    r = client.post("/api/training/start", json={
        "project_id": pid, "data_path": _jsonl(tmp_path), "model_path": "m", "output_dir": "output/mine",
    })
    assert r.status_code == 200, r.text
    assert fake_engine.started["config"].output_dir == "output/mine"


def test_bake_uses_project_system_prompt_when_form_sends_none(
    client, fake_engine: _FakeEngine, tmp_path: Path,
) -> None:
    pid = _project(client, "You are a Velmaris historian.")
    r = client.post("/api/training/start", json={
        "project_id": pid, "data_path": _jsonl(tmp_path), "model_path": "m", "system_prompt_mode": "bake",
    })
    assert r.status_code == 200, r.text
    assert fake_engine.started["system_prompt"] == "You are a Velmaris historian."


def test_mode_none_keeps_prompt_empty(client, fake_engine: _FakeEngine, tmp_path: Path) -> None:
    pid = _project(client, "You are a Velmaris historian.")
    r = client.post("/api/training/start", json={
        "project_id": pid, "data_path": _jsonl(tmp_path), "model_path": "m", "system_prompt_mode": "none",
    })
    assert r.status_code == 200, r.text
    assert fake_engine.started["system_prompt"] == ""
