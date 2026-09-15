"""Regression tests for testing-page auto-load (QABUG-007 / QABUG-011)."""

from __future__ import annotations

import json
import uuid
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock

import pytest
from fastapi.testclient import TestClient

from finetune_studio import db
from finetune_studio.config import settings
from finetune_studio.webui import app as app_module
from finetune_studio.webui.app import app


@pytest.fixture
def client_and_db(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    db_path = tmp_path / "fts_test.db"
    monkeypatch.setattr(settings, "db_path", str(db_path))
    db.init_db()
    return TestClient(app), db_path


def test_testing_auto_loads_merged_model(
    client_and_db: tuple[TestClient, Path],
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    client, _db_path = client_and_db
    r = client.post(
        "/api/projects",
        json={"name": f"auto-{uuid.uuid4().hex[:6]}", "base_model": "x/test"},
    )
    assert r.status_code == 200, r.text
    pid = r.json()["id"]

    out = tmp_path / "some" / "path"
    merged = out / "merged"
    merged.mkdir(parents=True)
    (merged / "config.json").write_text("{}", encoding="utf-8")

    run = db.create_run(pid, "done-run", base_model="x/test")
    db.update_run(run["id"], status="completed", output_path=str(out))

    load_calls: list[str] = []

    class FakeEngine:
        def __init__(self) -> None:
            self.model = None
            self.model_path = None
            self.is_gguf = False

        def load(self, path: str, **_kwargs: Any) -> None:
            load_calls.append(path)
            self.model = object()
            self.model_path = path

        def unload(self) -> None:
            self.model = None

        def generate(self, messages: list, **_kwargs: Any) -> str:
            return "Paris"

    fake = FakeEngine()
    monkeypatch.setattr(app_module, "inference_engine", fake)
    monkeypatch.setattr(
        "finetune_studio.webui.routes.testing.inference_engine",
        fake,
    )

    suite = [
        {
            "name": "q1",
            "category": "geo",
            "question": "Capital of France?",
            "correct_answer": "Paris",
            "keywords": ["Paris"],
        }
    ]
    suite_path = tmp_path / "suite.json"
    suite_path.write_text(json.dumps(suite), encoding="utf-8")

    resp = client.post(
        "/api/testing/run-suite",
        json={
            "suite_path": str(suite_path),
            "project_id": pid,
            "max_tokens": 32,
        },
    )
    assert resp.status_code == 200, resp.text
    assert load_calls, "expected inference_engine.load to be called"
    assert load_calls[0] == str(merged)


def test_testing_returns_400_when_no_completed_run(
    client_and_db: tuple[TestClient, Path],
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    client, _db_path = client_and_db
    r = client.post(
        "/api/projects",
        json={"name": f"empty-{uuid.uuid4().hex[:6]}", "base_model": "x/test"},
    )
    assert r.status_code == 200, r.text
    pid = r.json()["id"]

    fake = MagicMock()
    fake.model = None
    fake.model_path = None
    fake.is_gguf = False
    monkeypatch.setattr(app_module, "inference_engine", fake)
    monkeypatch.setattr(
        "finetune_studio.webui.routes.testing.inference_engine",
        fake,
    )

    suite_path = tmp_path / "suite.json"
    suite_path.write_text("[]", encoding="utf-8")

    resp = client.post(
        "/api/testing/run-suite",
        json={"suite_path": str(suite_path), "project_id": pid},
    )
    assert resp.status_code == 400, resp.text
    assert "no completed training run" in resp.json()["error"]
