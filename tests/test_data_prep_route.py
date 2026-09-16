"""Regression test for QABUG-016: the /projects/{pid}/data-prep page route
handler was missing, causing the URL to fall through to the SPA's 404
fallback (which returns only the base layout — ~474 chars, 10 refs —
without the main content). The data_prep.html template existed but had no
mounted route, so the page silently rendered as a near-empty stub."""

from __future__ import annotations

import pytest
from starlette.testclient import TestClient

from finetune_studio import db


@pytest.fixture
def client(monkeypatch, tmp_path):
    """Isolated FastAPI TestClient backed by a temp SQLite db + temp fs root.

    Mirrors the pattern in tests/test_dataset_register.py — points the app's
    Settings.db_path at a tmp file so the test never touches the real fan-dragon
    state.db. The Settings.fs_root is monkey-patched so file-library helpers
    route to a tmp dir.
    """
    db_path = tmp_path / "finetune_studio.db"
    fs_root = tmp_path / "fs"
    fs_root.mkdir()

    monkeypatch.setenv("FTS_TEST_DB_PATH", str(db_path))
    monkeypatch.setenv("FTS_TEST_FS_ROOT", str(fs_root))

    # Re-init the app's settings singleton to pick up the tmp paths.
    from finetune_studio import config as cfg_mod

    cfg = cfg_mod.Settings()
    cfg.db_path = str(db_path)
    cfg.fs_root = str(fs_root)
    # Re-bind the module-level Settings so app code reads the patched one.
    monkeypatch.setattr(cfg_mod, "settings", cfg)

    # Reset the module-level db module's cached connection so it re-opens on
    # the tmp db.
    import importlib

    importlib.reload(db)

    # Re-bind the routes' Settings imports to point at the patched singleton.
    # (FastAPI captured Settings at import time; we need a fresh app instance.)
    import finetune_studio.webui.app as app_mod
    importlib.reload(app_mod)
    yield TestClient(app_mod.app)


def _make_project(client) -> str:
    """Create a project + return its id (uses the real /api/projects route)."""
    r = client.post(
        "/api/projects",
        json={
            "name": "qabug-016-data-prep",
            "description": "regression fixture",
            "base_model": "",
            "system_prompt": "test",
        },
    )
    assert r.status_code == 200, r.text
    return r.json()["id"]


def test_data_prep_route_returns_200(client):
    """GET /projects/{pid}/data-prep returns 200 — not 404 (the bug)."""
    pid = _make_project(client)
    r = client.get(f"/projects/{pid}/data-prep")
    assert r.status_code == 200, (
        f"Expected 200, got {r.status_code}. "
        f"QABUG-016: missing route handler in pages.py — fix is the new "
        f"`project_data_prep_page` handler added between `project_data_page` "
        f"and `project_training_page`. Body (first 400 chars): {r.text[:400]}"
    )


def test_data_prep_route_renders_main_content(client):
    """Response body contains the data_prep.html template content (not just the base layout)."""
    pid = _make_project(client)
    r = client.get(f"/projects/{pid}/data-prep")
    assert r.status_code == 200
    body = r.text
    # The data_prep.html template has these anchors in its main content.
    assert "Uploaded files" in body, (
        "Body is missing the Uploaded files section — the route is rendering "
        "the SPA 404 fallback instead of data_prep.html. See QABUG-016."
    )
    assert "Parsed sources" in body, (
        "Body is missing the Parsed sources section — same QABUG-016 root cause."
    )
    assert "Run a prep job" in body, (
        "Body is missing the Run-a-prep-job form — same QABUG-016 root cause."
    )
    assert "Training / Q&amp;A output" in body or "Training / Q&A output" in body, (
        "Body is missing the Training / Q&A output list — same QABUG-016 root cause."
    )
    # The prep form should be wired to /api/projects/{pid}/data-prep/start.
    assert f"/api/projects/{pid}/data-prep/start" in body, (
        "Prep form is missing the project-scoped API endpoint. The page "
        "should POST to the project's data-prep/start route."
    )


def test_data_prep_route_renders_sources_or_empty(client):
    """The Source file dropdown iterates `sources or []` — empty list is fine
    for a fresh project, but the select element must render."""
    pid = _make_project(client)
    r = client.get(f"/projects/{pid}/data-prep")
    assert r.status_code == 200
    body = r.text
    assert '<select id="prep-source"' in body, (
        "Source file dropdown is missing from the rendered page. "
        "QABUG-016 root cause: route handler was missing."
    )
    assert 'pick a parsed source' in body.lower(), (
        "Source file dropdown placeholder is missing."
    )
