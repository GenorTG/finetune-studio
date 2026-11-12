"""Tests for "parse this library file now" (``POST /data-prep/sources {file_id}``).

History: QABUG-003 (library uploads never reached the pairs page's source list) and E2E-17 (promote must parse so
read_source finds the text). Since 2026-10-14 a source IS a library file (``knowledge.sources``): every readable
upload is parsed on arrival, and this route re-runs a parse that is missing or failed. Paths are never accepted.
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
    """Route the DB + FTS filesystem into temp dirs."""
    db_path = tmp_path / "fts_test.db"
    monkeypatch.setattr(settings, "db_path", str(db_path))
    root = tmp_path / "fts_root"
    projects = root / "projects"
    projects.mkdir(parents=True)
    monkeypatch.setattr("finetune_studio.data.fs.paths._ROOT", root)
    monkeypatch.setattr("finetune_studio.data.fs.paths._PROJECTS", projects)
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


def _register_uploaded_file(
    client, pid: str, contents: bytes, name: str = "sample.txt"
) -> str:
    r = client.post(
        f"/api/projects/{pid}/files/upload",
        files={"files": (name, contents, "text/plain")},
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
    assert source["file_id"] == fid
    assert os.path.exists(source["raw_path"])

    listed = client.get(f"/api/projects/{pid}/data-prep/sources").json()["sources"]
    assert any(s["id"] == source["id"] for s in listed)


def test_promote_parses_markdown_and_read_source_returns_text(client_and_db):
    """E2E-17: promote must parse+chunk; read_source must see file text."""
    client, db_path = client_and_db
    pid = _create_project(client, db_path)
    body_text = (
        "# Velmaris Worldbuilding\n\n"
        "The Ember College sits on the northern cliffs. " * 40
    )
    fid = _register_uploaded_file(
        client, pid, body_text.encode("utf-8"), name="velmaris_worldbuilding.md"
    )

    r = client.post(
        f"/api/projects/{pid}/data-prep/sources",
        json={"file_id": fid},
    )
    assert r.status_code == 200, r.text
    source = r.json()["source"]
    assert source.get("status") == "ready"
    assert int(source.get("chunk_count") or 0) > 0
    assert source.get("parser")

    listed = client.get(f"/api/projects/{pid}/data-prep/sources").json()["sources"]
    match = next(s for s in listed if s["id"] == source["id"])
    assert int(match.get("chunk_count") or 0) > 0

    from finetune_studio.data.fs.paths import file_dir
    from tests._guide_tool import run_tool as _run_tool

    sha = source["sha256"]
    parsed = file_dir(pid, sha) / "parsed.txt"
    assert parsed.is_file(), f"parsed.txt missing at {parsed}"
    assert "Ember College" in parsed.read_text(encoding="utf-8")

    tool = _run_tool(pid, "read_source", {"source_id": source["id"]})
    assert "Ember College" in (tool.get("text") or ""), tool


def test_promote_never_takes_a_path(client_and_db):
    """Decision 2026-10-14: files enter only through the file library; a ``data_path`` body is refused."""
    client, db_path = client_and_db
    pid = _create_project(client, db_path)
    with tempfile.NamedTemporaryFile(suffix=".txt", delete=False) as f:
        f.write(b"external content that must never be ingested.\n")
        path = f.name
    try:
        r = client.post(f"/api/projects/{pid}/data-prep/sources", json={"data_path": path})
        assert r.status_code == 400, r.text
        assert client.get(f"/api/projects/{pid}/data-prep/sources").json()["sources"] == []
    finally:
        os.unlink(path)


def test_promote_cross_project_file_id_is_404(client_and_db):
    client, db_path = client_and_db
    pid_a = _create_project(client, db_path)
    pid_b = _create_project(client, db_path)
    fid_a = _register_uploaded_file(client, pid_a, b"secret-a\n")

    r = client.post(
        f"/api/projects/{pid_b}/data-prep/sources",
        json={"file_id": fid_a},
    )
    assert r.status_code == 404, r.text
    assert "not found" in r.json()["detail"]


def test_promote_empty_body_is_400(client_and_db):
    client, db_path = client_and_db
    pid = _create_project(client, db_path)
    r = client.post(f"/api/projects/{pid}/data-prep/sources", json={})
    assert r.status_code == 400, r.text
    assert "required" in r.json()["detail"]


def test_promote_is_idempotent(client_and_db):
    client, db_path = client_and_db
    pid = _create_project(client, db_path)
    fid = _register_uploaded_file(
        client, pid, b"idempotent promote text long enough for parse.\n"
    )

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
    assert int(r1.json()["source"].get("chunk_count") or 0) > 0
    assert int(r2.json()["source"].get("chunk_count") or 0) > 0
