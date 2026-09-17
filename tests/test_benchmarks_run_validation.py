"""Tests for benchmark run validation (E2E-41 / E2E-43)."""

from __future__ import annotations

import json
import uuid
from pathlib import Path
from typing import Any, ClassVar
from unittest.mock import MagicMock

import pytest
from fastapi.testclient import TestClient

from finetune_studio import db
from finetune_studio.config import settings
from finetune_studio.webui.app import app


@pytest.fixture
def client_and_db(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    db_path = tmp_path / "fts_bench_val.db"
    monkeypatch.setattr(settings, "db_path", str(db_path))
    db.init_db()
    return TestClient(app), db_path


def _create_project(client: TestClient) -> str:
    r = client.post(
        "/api/projects",
        json={"name": f"bench-val-{uuid.uuid4().hex[:6]}", "base_model": "org/base"},
    )
    assert r.status_code == 200, r.text
    return r.json()["id"]


def _write_suite(tmp_path: Path) -> str:
    path = tmp_path / "suite.json"
    path.write_text(
        json.dumps(
            [
                {
                    "name": "capital_fr",
                    "category": "geo",
                    "question": "Capital of France?",
                    "correct_answer": "Paris",
                    "keywords": ["Paris"],
                }
            ]
        ),
        encoding="utf-8",
    )
    return str(path)


class _TrackingEngine:
    """Mock InferenceEngine that records load/unload calls."""

    instances: ClassVar[list[_TrackingEngine]] = []

    def __init__(self) -> None:
        self.model: object | None = None
        self.model_path: str | None = None
        self.load_calls: list[str] = []
        self.unload_calls: int = 0
        _TrackingEngine.instances.append(self)

    def load(self, path: str, **_kwargs: Any) -> None:
        self.load_calls.append(path)
        self.model = object()
        self.model_path = path

    def unload(self) -> None:
        self.unload_calls += 1
        self.model = None

    def generate(self, messages: list, **_kwargs: Any) -> str:
        return "Paris"


def _install_tracking_engine(
    monkeypatch: pytest.MonkeyPatch,
    *,
    global_model: object | None = object(),
) -> MagicMock:
    _TrackingEngine.instances.clear()
    monkeypatch.setattr(
        "finetune_studio.testing.inference.InferenceEngine",
        _TrackingEngine,
    )
    global_ie = MagicMock()
    global_ie.model = global_model
    global_ie.unload = MagicMock()
    monkeypatch.setattr(
        "finetune_studio.webui.app.inference_engine",
        global_ie,
        raising=False,
    )
    return global_ie


def test_missing_suite_returns_404_without_loading(
    client_and_db: tuple[TestClient, Path],
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    client, _ = client_and_db
    pid = _create_project(client)
    out = tmp_path / "out"
    out.mkdir()
    (out / "merged").mkdir()
    (out / "merged" / "config.json").write_text("{}", encoding="utf-8")
    run = db.create_run(pid, "train-ok", base_model="org/base")
    db.update_run(run["id"], status="done", output_path=str(out))

    global_ie = _install_tracking_engine(monkeypatch, global_model=object())

    r = client.post(
        f"/api/benchmarks/projects/{pid}/runs/{run['id']}/run",
        json={
            "suite_path": str(tmp_path / "does_not_exist.json"),
            "suite_name": "missing",
            "judge_mode": "heuristic",
        },
    )
    assert r.status_code == 404, r.text
    body = r.json()
    assert "error" in body
    assert "not found" in body["error"].lower() or "suite" in body["error"].lower()
    assert _TrackingEngine.instances == []
    global_ie.unload.assert_not_called()


def test_failed_run_returns_409(
    client_and_db: tuple[TestClient, Path],
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    client, _ = client_and_db
    pid = _create_project(client)
    run = db.create_run(pid, "train-fail", base_model="org/base")
    db.update_run(
        run["id"],
        status="failed",
        error="OOM during train",
        output_path="",
    )
    suite_path = _write_suite(tmp_path)
    global_ie = _install_tracking_engine(monkeypatch)

    r = client.post(
        f"/api/benchmarks/projects/{pid}/runs/{run['id']}/run",
        json={
            "suite_path": suite_path,
            "suite_name": "kw",
            "judge_mode": "heuristic",
        },
    )
    assert r.status_code == 409, r.text
    err = r.json()["error"]
    assert "no trained model" in err
    assert "failed" in err
    assert _TrackingEngine.instances == []
    global_ie.unload.assert_not_called()


def test_done_run_loads_and_unloads_even_when_run_suite_raises(
    client_and_db: tuple[TestClient, Path],
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    client, _ = client_and_db
    pid = _create_project(client)
    out = tmp_path / "out2"
    out.mkdir()
    (out / "merged").mkdir()
    (out / "merged" / "config.json").write_text("{}", encoding="utf-8")
    run = db.create_run(pid, "train-ok", base_model="org/base")
    db.update_run(run["id"], status="done", output_path=str(out))
    suite_path = _write_suite(tmp_path)

    global_ie = _install_tracking_engine(monkeypatch)

    def _boom(*_a: Any, **_k: Any) -> list:
        raise RuntimeError("suite exploded")

    monkeypatch.setattr(
        "finetune_studio.testing.suite.run_suite",
        _boom,
    )

    r = client.post(
        f"/api/benchmarks/projects/{pid}/runs/{run['id']}/run",
        json={
            "suite_path": suite_path,
            "suite_name": "kw",
            "judge_mode": "heuristic",
            "max_tokens": 16,
        },
    )
    assert r.status_code == 500, r.text
    assert "error" in r.json()
    assert len(_TrackingEngine.instances) == 1
    eng = _TrackingEngine.instances[0]
    assert eng.load_calls, "expected InferenceEngine.load to be called"
    assert eng.unload_calls >= 1, "expected unload in finally"
    global_ie.unload.assert_called()


def test_done_run_accepts_explicit_export_and_records_target(
    client_and_db: tuple[TestClient, Path],
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    client, _ = client_and_db
    pid = _create_project(client)
    out = tmp_path / "out3"
    out.mkdir()
    (out / "merged").mkdir()
    (out / "merged" / "config.json").write_text("{}", encoding="utf-8")
    export = out / "model-q8_0.gguf"
    export.write_bytes(b"GGUF")
    run = db.create_run(pid, "train-ok", base_model="org/base")
    db.update_run(run["id"], status="done", output_path=str(out))
    suite_path = _write_suite(tmp_path)
    _install_tracking_engine(monkeypatch, global_model=None)

    r = client.post(
        f"/api/benchmarks/projects/{pid}/runs/{run['id']}/run",
        json={"suite_path": suite_path, "model_path": str(export)},
    )
    assert r.status_code == 200, r.text
    assert r.json()["benchmark"]["scores"]["model_path"] == str(export)
    assert _TrackingEngine.instances[0].load_calls == [str(export)]
