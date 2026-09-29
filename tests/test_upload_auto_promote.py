"""Parser-supported uploads are staged immediately and parsed in background."""

from __future__ import annotations

import uuid

import pytest
from fastapi.testclient import TestClient

from finetune_studio import db
from finetune_studio.config import settings
from finetune_studio.webui.app import app


@pytest.fixture
def client_and_db(tmp_path, monkeypatch):
    db_path = tmp_path / "fts_test.db"
    monkeypatch.setattr(settings, "db_path", str(db_path))
    root = tmp_path / "fts_root"
    projects = root / "projects"
    projects.mkdir(parents=True)
    monkeypatch.setattr("finetune_studio.data.fs.paths._ROOT", root)
    monkeypatch.setattr("finetune_studio.data.fs.paths._PROJECTS", projects)
    db.init_db()
    return TestClient(app), db_path


def _create_project(client) -> str:
    r = client.post(
        "/api/projects",
        json={"name": f"auto-{uuid.uuid4().hex[:6]}", "base_model": "x/test"},
    )
    assert r.status_code == 200, r.text
    return r.json()["id"]


@pytest.mark.parametrize(
    "name,body",
    [
        ("notes.txt", b"plain text long enough to parse into at least one chunk.\n"),
        ("doc.md", b"# Title\n\nMarkdown body long enough to parse into chunks.\n" * 5),
        ("doc.markdown", b"# Markdown\n\nBody text long enough for the text parser.\n" * 5),
        ("app.log", b"INFO start\nINFO more log lines for parsing into chunks.\n" * 8),
    ],
)
def test_text_upload_auto_promotes_to_parsed_source(client_and_db, name, body):
    client, _ = client_and_db
    pid = _create_project(client)

    r = client.post(
        f"/api/projects/{pid}/files/upload",
        files={"files": (name, body, "text/plain")},
    )
    assert r.status_code == 200, r.text
    report = r.json()["report"]
    assert len(report) == 1
    item = report[0]
    assert item["status"] == "uploaded"
    assert item.get("source_id"), item
    assert item.get("source"), item
    assert item["source"].get("status") == "queued"
    assert item.get("parse_status") == "queued"

    # TestClient waits for the response's background tasks, so the durable
    # source has completed parsing even though the response snapshot says queued.
    listed = client.get(f"/api/projects/{pid}/data-prep/sources").json()["sources"]
    parsed = next(s for s in listed if s["id"] == item["source_id"])
    assert parsed["status"] == "ready"
    assert int(parsed.get("chunk_count") or 0) > 0


def test_parser_supported_binary_is_staged_and_parse_failure_is_visible(client_and_db):
    client, _ = client_and_db
    pid = _create_project(client)
    # Tiny PNG-ish bytes with a non-text extension.
    r = client.post(
        f"/api/projects/{pid}/files/upload",
        files={"files": ("photo.png", b"\x89PNG\r\n\x1a\nnot-really", "image/png")},
    )
    assert r.status_code == 200, r.text
    item = r.json()["report"][0]
    assert item["status"] == "uploaded"
    assert item.get("source_id")
    assert item.get("parse_status") == "queued"
    listed = client.get(f"/api/projects/{pid}/data-prep/sources").json()["sources"]
    assert len(listed) == 1
    assert listed[0]["stage"] == "error"
    assert listed[0]["error"]


def test_duplicate_text_upload_still_ensures_source(client_and_db):
    client, _ = client_and_db
    pid = _create_project(client)
    payload = b"duplicate promote content that is long enough to parse.\n" * 3
    r1 = client.post(
        f"/api/projects/{pid}/files/upload",
        files={"files": ("dup.txt", payload, "text/plain")},
    )
    assert r1.status_code == 200, r1.text
    src_id = r1.json()["report"][0]["source_id"]

    r2 = client.post(
        f"/api/projects/{pid}/files/upload",
        files={"files": ("dup.txt", payload, "text/plain")},
    )
    assert r2.status_code == 200, r2.text
    item = r2.json()["report"][0]
    assert item["status"] == "duplicate"
    assert item.get("source_id") == src_id
