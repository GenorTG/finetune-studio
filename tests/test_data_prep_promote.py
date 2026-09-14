"""Tests for data-prep source promote-from-library (QABUG-003 regression).

QABUG-003 was: file-library uploads land in project_files, but the data-prep
"Source file" picker reads pfs.list_qa_sources(pid) — a different store —
so freshly uploaded files never appeared until a second upload through
the prep pipeline.
"""

from __future__ import annotations

import os
import tempfile
import uuid

import pytest
from fastapi.testclient import TestClient

from finetune_studio import db
from finetune_studio.config import settings
from finetune_studio.webui.app import app


@pytest.fixture
def client_and_db(tmp_path, monkeypatch):
    """Route the DB into a temp file so tests don't touch fan-dragon state."""
    db_path = tmp_path / "fts_test.db"
    monkeypatch.setattr(settings, "db_path", str(db_path))
    db.init_db()
    client = TestClient(app)
    return client, db_path


def _create_project(client, db_path: object) -> str:
    r = client.post(
        "/api/projects",
        json={"name": f"qa-{uuid.uuid4().hex[:6]}", "base_model": "x/test"},
    )
    assert r.status_code == 200, r.text
    return r.json()["id"]


def _register_uploaded_file(client, pid: str, contents: bytes) -> str:
    r = client.post(
        f"/api/projects/{pid}/files/upload",
        files={"files": ("sample.txt", contents, "text/plain")},
    )
    assert r.status_code == 200, r.text
    return r.json()["report"][0]["file_id"]


def test_promote_with_file_id_resolves_and_returns_source(client_and_db):
    client, db_path = client_and_db
    pid = _create_project(client, db_path)
    fid = _register_uploaded_file(client, pid, b"hello promote world\n")

    r = client.post(
        f"/api/projects/{pid}/data-prep/sources",
        json={"file_id": fid},
    )
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["ok"] is True
    source = body["source"]
    assert source["id"]
    assert source["filename"]
    assert source.get("data_path") or source.get("path")
    assert os.path.exists(source.get("data_path") or source.get("path"))

    listed = client.get(f"/api/projects/{pid}/data-prep/sources").json()["sources"]
    assert any(s["id"] == source["id"] for s in listed)


def test_promote_with_data_path_directly(client_and_db):
    client, db_path = client_and_db
    pid = _create_project(client, db_path)
    with tempfile.NamedTemporaryFile(suffix=".txt", delete=False) as f:
        f.write(b"direct path promote\n")
        path = f.name
    try:
        r = client.post(
            f"/api/projects/{pid}/data-prep/sources",
            json={"data_path": path},
        )
        assert r.status_code == 200, r.text
        source = r.json()["source"]
        assert source["data_path"] == os.path.realpath(path) or source["path"] == path
    finally:
        os.unlink(path)


def test_promote_cross_project_file_id_is_403(client_and_db):
    client, db_path = client_and_db
    pid_a = _create_project(client, db_path)
    pid_b = _create_project(client, db_path)
    fid_a = _register_uploaded_file(client, pid_a, b"secret-a\n")

    r = client.post(
        f"/api/projects/{pid_b}/data-prep/sources",
        json={"file_id": fid_a},
    )
    assert r.status_code == 403, r.text
    assert "another project" in r.json()["error"]


def test_promote_empty_body_is_400(client_and_db):
    client, db_path = client_and_db
    pid = _create_project(client, db_path)
    r = client.post(f"/api/projects/{pid}/data-prep/sources", json={})
    assert r.status_code == 400, r.text
    assert "required" in r.json()["error"]


def test_promote_is_idempotent(client_and_db):
    client, db_path = client_and_db
    pid = _create_project(client, db_path)
    fid = _register_uploaded_file(client, pid, b"idempotent promote\n")

    r1 = client.post(
        f"/api/projects/{pid}/data-prep/sources",
        json={"file_id": fid},
    )
    r2 = client.post(
        f"/api/projects/{pid}/data-prep/sources",
        json={"file_id": fid},
    )
    assert r1.status_code == 200, r1.text
    assert r2.status_code == 200, r2.text
    assert r1.json()["source"]["id"] == r2.json()["source"]["id"]
