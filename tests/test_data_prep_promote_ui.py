"""Data Prep UI: promote older library text/markdown into Source file picker.

Browser gap: indexed .txt/.md files appear in the file library (and may even
show a parsed-MD indicator) while the Source file <select> stays empty because
pre-auto-promote uploads were never registered as QA sources, and there was
no visible action to promote them.
"""

from __future__ import annotations

import uuid
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from finetune_studio import db
from finetune_studio.config import settings
from finetune_studio.data.fs.qa import AUTO_PROMOTE_EXTENSIONS
from finetune_studio.webui.app import app

_TEMPLATE = (
    Path(__file__).resolve().parents[1]
    / "src"
    / "finetune_studio"
    / "webui"
    / "templates"
    / "data_prep.html"
)


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


def _create_project(client: TestClient) -> str:
    r = client.post(
        "/api/projects",
        json={"name": f"promo-ui-{uuid.uuid4().hex[:6]}", "base_model": "x/test"},
    )
    assert r.status_code == 200, r.text
    return r.json()["id"]


def test_template_exposes_use_as_source_action() -> None:
    src = _TEMPLATE.read_text(encoding="utf-8")
    assert "Use as source" in src
    assert "flUseAsSource" in src
    assert "flPromoteEligibleFromLibrary" in src
    assert "Use text/markdown from file library" in src
    assert "prep-promote-panel" in src
    # Promote helpers must surface plain status text, not dump response JSON.
    promote_js = src.split("async function flPromoteToDataPrep", 1)[1].split(
        "function flShowUploadReport", 1
    )[0]
    assert "JSON.stringify(d" not in promote_js
    assert "JSON.stringify(result" not in promote_js
    assert "Debug · raw JSON" not in promote_js
    assert "flReport(" in promote_js


def test_template_promote_exts_match_server() -> None:
    src = _TEMPLATE.read_text(encoding="utf-8")
    assert "auto_promote_extensions" in src
    for ext in AUTO_PROMOTE_EXTENSIONS:
        assert ext.startswith(".")


def test_data_prep_page_injects_promote_extensions(client_and_db) -> None:
    client, _ = client_and_db
    pid = _create_project(client)
    r = client.get(f"/projects/{pid}/data-prep")
    assert r.status_code == 200, r.text
    body = r.text
    assert "Use as source" in body
    assert "Use text/markdown from file library" in body
    assert "prep-promote-panel" in body
    for ext in sorted(AUTO_PROMOTE_EXTENSIONS):
        assert f'"{ext}"' in body or f"'{ext}'" in body
    # Source picker block must not dump a raw sources JSON blob as the UI.
    picker = body.split('id="prep-source"', 1)[1].split("id=\"prep-qpc\"", 1)[0]
    assert "JSON.stringify" not in picker
    assert '"ok": true' not in picker


def test_older_upload_without_auto_promote_then_manual_promote(client_and_db, monkeypatch) -> None:
    """Simulate a pre-auto-promote library file: in project_files, not in sources."""
    client, _ = client_and_db
    pid = _create_project(client)

    monkeypatch.setattr(
        "finetune_studio.data.fs.qa.maybe_auto_promote_upload",
        lambda *a, **k: None,
    )
    body = b"# Older notes\n\n" + (b"Chunkable markdown body. " * 40)
    up = client.post(
        f"/api/projects/{pid}/files/upload",
        files={"files": ("older_notes.md", body, "text/markdown")},
    )
    assert up.status_code == 200, up.text
    item = up.json()["report"][0]
    assert item["status"] == "uploaded"
    assert not item.get("source_id")
    fid = item["file_id"]

    listed_before = client.get(f"/api/projects/{pid}/data-prep/sources").json()["sources"]
    assert listed_before == []

    promo = client.post(
        f"/api/projects/{pid}/data-prep/sources",
        json={"file_id": fid},
    )
    assert promo.status_code == 200, promo.text
    payload = promo.json()
    assert payload["ok"] is True
    source = payload["source"]
    assert source["filename"] == "older_notes.md"
    assert source.get("status") == "ready"
    assert int(source.get("chunk_count") or 0) > 0

    listed = client.get(f"/api/projects/{pid}/data-prep/sources").json()["sources"]
    assert any(s["id"] == source["id"] for s in listed)

    page = client.get(f"/projects/{pid}/data-prep")
    assert page.status_code == 200
    assert source["id"] in page.text
    assert "older_notes.md" in page.text
