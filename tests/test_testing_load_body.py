"""Regression: POST /api/testing/load accepts path|model_path (QABUG-014-runtime)."""

from __future__ import annotations

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
def client_and_engine(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> tuple[TestClient, MagicMock]:
    db_path = tmp_path / "fts_test.db"
    monkeypatch.setattr(settings, "db_path", str(db_path))
    db.init_db()

    fake = MagicMock()
    fake.model = None
    fake.model_path = None
    fake.is_gguf = False
    fake.vision = False

    def _load(path: str, **kwargs: Any) -> None:
        fake.model = object()
        fake.model_path = path
        fake.last_kwargs = kwargs

    fake.load.side_effect = _load
    monkeypatch.setattr(app_module, "inference_engine", fake)
    monkeypatch.setattr(
        "finetune_studio.webui.routes.testing.inference_engine",
        fake,
    )
    return TestClient(app), fake


def test_testing_load_accepts_model_path(
    client_and_engine: tuple[TestClient, MagicMock],
) -> None:
    client, fake = client_and_engine
    resp = client.post("/api/testing/load", json={"model_path": "/models/a"})
    assert resp.status_code == 200, resp.text
    assert resp.json()["status"] == "loaded"
    assert resp.json()["model"] == "/models/a"
    fake.load.assert_called_once()
    assert fake.load.call_args.args[0] == "/models/a"


def test_testing_load_accepts_path_alias(
    client_and_engine: tuple[TestClient, MagicMock],
) -> None:
    client, fake = client_and_engine
    resp = client.post("/api/testing/load", json={"path": "/models/b"})
    assert resp.status_code == 200, resp.text
    assert resp.json()["model"] == "/models/b"
    assert fake.load.call_args.args[0] == "/models/b"


def test_testing_load_rejects_empty_body(
    client_and_engine: tuple[TestClient, MagicMock],
) -> None:
    client, fake = client_and_engine
    resp = client.post("/api/testing/load", json={})
    assert resp.status_code == 200, resp.text
    assert resp.json()["error"] == "No model_path"
    fake.load.assert_not_called()


def test_chat_v2_load_accepts_path_alias(
    client_and_engine: tuple[TestClient, MagicMock],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    client, fake = client_and_engine
    # chat_v2 imports inference_engine from app inside the handler
    monkeypatch.setattr(app_module, "inference_engine", fake)
    resp = client.post("/api/chat-v2/load", json={"path": "/models/c"})
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert body.get("error") is None, body
    assert body["model"] == "/models/c"
    assert fake.load.call_args.args[0] == "/models/c"
