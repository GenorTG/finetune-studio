"""Data Prep UI: "Use as source" parses a library file that has no parse yet (or a failed one).

Every readable upload is parsed on arrival (``knowledge.sources``); the button covers files whose parse never ran
(an upload that predates auto-parsing) or failed, and it is offered for every parser-supported extension.
"""

from __future__ import annotations

import uuid
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from finetune_studio import db
from finetune_studio.config import settings
from finetune_studio.data.parsers import readable_extensions
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
    assert "promotable_extensions | tojson" in src


def test_data_prep_page_injects_promote_extensions(client_and_db) -> None:
    client, _ = client_and_db
    pid = _create_project(client)
    r = client.get(f"/projects/{pid}/data-prep")
    assert r.status_code == 200, r.text
    body = r.text
    assert "Use as source" in body
    assert "Use text/markdown from file library" in body
    assert 'id="dp-promote-panel"' in body
    for ext in (".txt", ".md", ".pdf", ".docx"):
        assert ext in readable_extensions() and f'"{ext}"' in body
    # The files table is rendered by script from the API, never a raw sources JSON blob in the page.
    host = body.split('id="dp-files"', 1)[1].split("</div>", 1)[0]
    assert "JSON.stringify" not in host and '"ok": true' not in host


def test_older_upload_without_auto_promote_then_manual_promote(client_and_db, monkeypatch) -> None:
    """Simulate a library file whose parse never ran: in project_files, no source yet."""
    client, _ = client_and_db
    pid = _create_project(client)

    from finetune_studio.webui.routes import file_library

    original_stage = file_library._stage_for_parse
    monkeypatch.setattr(file_library, "_stage_for_parse", lambda *a, **k: None)
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
    monkeypatch.setattr(file_library, "_stage_for_parse", original_stage)

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
    assert page.status_code == 200 and 'id="dp-files"' in page.text   # the files table renders from the sources API
