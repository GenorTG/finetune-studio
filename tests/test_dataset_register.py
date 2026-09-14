"""Tests for the dataset-register route (QABUG-001 regression).

QABUG-001 was: POST /api/projects/{pid}/datasets returned
{"error":"data_path required"} when the WebUI data-prep frontend sent
{file_id: "…"} instead of {data_path: "…"}.
"""

import os
import sqlite3
import tempfile
import uuid

import pytest
from fastapi.testclient import TestClient

from finetune_studio import db
from finetune_studio.config import settings
from finetune_studio.webui.app import app


@pytest.fixture
def client_and_db(tmp_path, monkeypatch):
    """Route the DB into a temp file so tests don't touch fan-dragon state.

    `db.init_db()` resolves `settings.db_path` from the singleton at call
    time, so we point the setting at the tmp file before init + before
    TestClient boots the app (which routes lifespan-startup into init_db()).
    """
    db_path = tmp_path / "fts_test.db"
    monkeypatch.setattr(settings, "db_path", str(db_path))
    db.init_db()
    # Reset the project_datasets and project_files tables in case a previous
    # test in this process leaked rows.
    client = TestClient(app)
    return client, db_path


def _create_project(client, db_path) -> str:
    """Helper: create a fresh project + return its pid."""
    r = client.post(
        "/api/projects",
        json={"name": f"qa-{uuid.uuid4().hex[:6]}", "base_model": "x/test"},
    )
    assert r.status_code == 200, r.text
    return r.json()["id"]


def _register_uploaded_file(client, pid: str, contents: bytes) -> str:
    """Helper: simulate a file upload and return the file id."""
    r = client.post(
        f"/api/projects/{pid}/files/upload",
        files={"files": ("sample.jsonl", contents, "application/octet-stream")},
    )
    assert r.status_code == 200, r.text
    return r.json()["report"][0]["file_id"]


def _cleanup_rows(db_path) -> None:
    """Wipe project-scoped rows between tests so ids don't collide."""
    con = sqlite3.connect(str(db_path))
    try:
        for tbl in (
            "project_datasets",
            "project_files",
            "projects",
        ):
            try:
                con.execute(f"DELETE FROM {tbl}")
            except sqlite3.OperationalError:
                pass
        con.commit()
    finally:
        con.close()


def test_register_accepts_file_id(client_and_db):
    """QABUG-001: when the UI sends {file_id}, the route must resolve stored_path."""
    client, db_path = client_and_db
    pid = _create_project(client, db_path)
    fid = _register_uploaded_file(client, pid, b'{"prompt":"hi","completion":"yo"}\n')

    r = client.post(f"/api/projects/{pid}/datasets", json={"file_id": fid})
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["project_id"] == pid
    assert body["data_path"]  # resolved from project_files
    assert os.path.exists(body["data_path"])
    # Listed back via GET /datasets.
    listed = client.get(f"/api/projects/{pid}/datasets").json()["datasets"]
    assert any(d["id"] == body["id"] for d in listed)


def test_register_accepts_data_path_directly(client_and_db):
    client, db_path = client_and_db
    pid = _create_project(client, db_path)
    with tempfile.NamedTemporaryFile(suffix=".jsonl", delete=False) as f:
        f.write(b'{"prompt":"a","completion":"b"}\n')
        path = f.name
    try:
        r = client.post(f"/api/projects/{pid}/datasets", json={"data_path": path})
        assert r.status_code == 200, r.text
        assert r.json()["data_path"] == path
    finally:
        os.unlink(path)


def test_register_rejects_when_neither_field_supplied(client_and_db):
    client, db_path = client_and_db
    pid = _create_project(client, db_path)
    r = client.post(f"/api/projects/{pid}/datasets", json={})
    assert r.status_code == 400, r.text
    assert "required" in r.json()["error"]


def test_register_file_id_from_other_project_is_403(client_and_db):
    """Cross-project file_id must NOT resolve to a different project's path."""
    client, db_path = client_and_db
    pid_a = _create_project(client, db_path)
    pid_b = _create_project(client, db_path)
    fid_a = _register_uploaded_file(client, pid_a, b'{"prompt":"x","completion":"y"}\n')

    r = client.post(f"/api/projects/{pid_b}/datasets", json={"file_id": fid_a})
    assert r.status_code == 403, r.text
    assert "another project" in r.json()["error"]


def test_register_file_id_unknown_is_400(client_and_db):
    client, db_path = client_and_db
    pid = _create_project(client, db_path)
    r = client.post(
        f"/api/projects/{pid}/datasets",
        json={"file_id": "deadbeefdeadbeefdeadbeefdeadbeef"},
    )
    assert r.status_code == 400, r.text  # resolved to "" then 400
    assert "required" in r.json()["error"]
