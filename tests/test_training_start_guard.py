"""Training-start hub-download guard (HANDOFF step 5).

A bare ``org/repo`` model_path used to make ``from_pretrained`` stream
multi-GB weights mid-run (stalled run c327fa36 attempt #1). The guard blocks
that unless the model is already local (hub cache / app hf_models dir) or the
caller explicitly opts in with ``allow_download=true``.
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
            status = "idle"

        self.state = _State()

    def on_update(self, cb: object) -> None:
        self.cb = cb

    def start(self, config: object, data: list, system_prompt: str, *, run_id: str, output_dir: str,
              project_id: str | None) -> str:
        self.started = {"config": config, "data": data, "system_prompt": system_prompt, "run_id": run_id,
                        "output_dir": output_dir, "project_id": project_id}
        return "training-job"


@pytest.fixture
def fake_engine(monkeypatch: pytest.MonkeyPatch) -> _FakeEngine:
    import finetune_studio.webui.routes.training as training_routes

    eng = _FakeEngine()
    monkeypatch.setattr(training_routes, "training_engine", eng)
    return eng


@pytest.fixture
def fake_home(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Point Path.home() at an empty sandbox so cache probes are deterministic."""
    home = tmp_path / "home"
    home.mkdir()
    monkeypatch.setattr(Path, "home", classmethod(lambda cls: home))
    return home


def _project(client) -> str:
    r = client.post("/api/projects", json={"name": "Guard", "system_prompt": "", "allow_duplicate": True})
    assert r.status_code in (200, 201), r.text
    return r.json()["id"]


def _jsonl(tmp_path: Path) -> str:
    p = tmp_path / "qa.jsonl"
    row = {"messages": [{"role": "user", "content": "q"}, {"role": "assistant", "content": "a"}]}
    p.write_text(json.dumps(row) + "\n", encoding="utf-8")
    return str(p)


def _start(client, pid: str, data_path: str, model_path: str, **extra):
    body = {"project_id": pid, "data_path": data_path, "model_path": model_path}
    body.update(extra)
    return client.post("/api/training/start", json=body)


def test_remote_repo_id_is_blocked(client, fake_engine, fake_home, tmp_path):
    pid = _project(client)
    r = _start(client, pid, _jsonl(tmp_path), "SomeOrg/BigModel-70B")
    body = r.json()
    assert "detail" in body and "not local" in body["detail"], body
    assert "allow_download" in body["detail"], body
    assert not fake_engine.started, "engine must not start when the guard fires"


def test_allow_download_proceeds(client, fake_engine, fake_home, tmp_path):
    pid = _project(client)
    r = _start(client, pid, _jsonl(tmp_path), "SomeOrg/BigModel-70B", allow_download=True)
    assert r.json().get("status") == "started", r.text
    assert fake_engine.started["config"].model_path == "SomeOrg/BigModel-70B"


def test_hub_cache_hit_passes_through(client, fake_engine, fake_home, tmp_path):
    snap = fake_home / ".cache" / "huggingface" / "hub" / "models--SomeOrg--BigModel-70B" / "snapshots" / "abc123"
    snap.mkdir(parents=True)
    (snap / "config.json").write_text("{}", encoding="utf-8")
    pid = _project(client)
    r = _start(client, pid, _jsonl(tmp_path), "SomeOrg/BigModel-70B")
    assert r.json().get("status") == "started", r.text
    # hub cache is used by transformers itself — id stays as-is
    assert fake_engine.started["config"].model_path == "SomeOrg/BigModel-70B"


def test_app_hf_models_dir_is_rewritten_to_local(client, fake_engine, fake_home, tmp_path):
    app_dir = fake_home / ".finetune-studio" / "hf_models" / "SomeOrg__BigModel-70B"
    app_dir.mkdir(parents=True)
    (app_dir / "config.json").write_text("{}", encoding="utf-8")
    pid = _project(client)
    r = _start(client, pid, _jsonl(tmp_path), "SomeOrg/BigModel-70B")
    assert r.json().get("status") == "started", r.text
    assert fake_engine.started["config"].model_path == str(app_dir)


def test_existing_local_path_untouched(client, fake_engine, fake_home, tmp_path):
    model_dir = tmp_path / "local-model"
    model_dir.mkdir()
    pid = _project(client)
    r = _start(client, pid, _jsonl(tmp_path), str(model_dir))
    assert r.json().get("status") == "started", r.text
    assert fake_engine.started["config"].model_path == str(model_dir)


def test_start_passes_the_run_to_the_job_and_a_busy_gpu_is_a_409(client, fake_engine, tmp_path, monkeypatch):
    """The route hands run id / output dir / project to the job process; a busy GPU is a real 409, not a 500."""
    from finetune_studio import db
    from finetune_studio.jobs.launch import JobRejected

    model_dir = tmp_path / "local-model"
    model_dir.mkdir()
    pid = _project(client)
    r = _start(client, pid, _jsonl(tmp_path), str(model_dir))
    assert r.status_code == 200, r.text
    run_id = r.json()["run_id"]
    assert fake_engine.started["run_id"] == run_id and fake_engine.started["project_id"] == pid
    assert fake_engine.started["output_dir"].endswith(f"runs/{run_id}")

    def busy(*a, **k):
        raise JobRejected(409, "the GPU is busy with export job x (export)", "gpu_busy")

    monkeypatch.setattr(fake_engine, "start", busy)
    r = _start(client, pid, _jsonl(tmp_path), str(model_dir))
    assert r.status_code == 409, r.text
    assert r.json()["code"] == "gpu_busy" and "GPU is busy" in r.json()["detail"]
    refused = [x for x in db.list_runs(pid) if x["id"] != run_id]
    assert [x["status"] for x in refused] == ["error"]
