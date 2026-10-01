"""Regression tests for the B1 (webui/routes core) documentation audit.

Pins three fixes made 2026-10-01:

1. ``projects.py`` run/rag sub-resource routes used to trust the ``{rid}``/
   ``{rid}`` path params without checking they belong to the ``{pid}`` in the
   URL — a run or rag id from a different project was fully readable and
   mutable through any other project's URL. Every such route now 404s for a
   foreign-owned or missing resource (see ``_get_owned_run`` /
   ``_get_owned_rag`` in ``routes/projects.py``).
2. ``routes/updates.py::get_update_status`` documented a ``?full=1`` query
   param in its docstring but never actually declared it on the route, so
   ``full`` was hardcoded ``False`` and the full log text was unreachable.
3. ``routes/data.py::upload_file`` joined the client-supplied filename
   directly into ``settings.data_dir`` without sanitizing it, allowing a
   crafted filename (e.g. ``../../x``) to write outside the data directory.
"""

from __future__ import annotations

import os

import pytest
from fastapi.testclient import TestClient

from finetune_studio import db
from finetune_studio.webui.app import app

MISSING_ID = "deadbeef"


@pytest.fixture(scope="module")
def client() -> TestClient:
    return TestClient(app, raise_server_exceptions=False)


@pytest.fixture()
def two_projects():
    """Two real projects, each with a run and a rag, for cross-project checks."""
    p1 = db.create_project(name="audit-b1-proj-1")
    p2 = db.create_project(name="audit-b1-proj-2")
    run2 = db.create_run(project_id=p2["id"], name="run-in-p2")
    rag2 = db.create_rag(project_id=p2["id"], name="rag-in-p2")
    return p1, p2, run2, rag2


# ── projects.py: cross-project run/rag ownership ──────────────────────────


def test_get_run_404s_for_run_owned_by_another_project(client, two_projects):
    p1, p2, run2, _rag2 = two_projects
    resp = client.get(f"/api/projects/{p1['id']}/runs/{run2['id']}")
    assert resp.status_code == 404, (
        f"run {run2['id']} belongs to {p2['id']}, not {p1['id']}; expected 404, "
        f"got {resp.status_code}: {resp.text[:200]}"
    )


def test_update_run_404s_for_foreign_run(client, two_projects):
    p1, _p2, run2, _rag2 = two_projects
    resp = client.patch(
        f"/api/projects/{p1['id']}/runs/{run2['id']}", json={"name": "hijacked"}
    )
    assert resp.status_code == 404
    # The run itself must be untouched.
    fresh = db.get_run(run2["id"])
    assert fresh["name"] == "run-in-p2"


def test_delete_run_404s_for_foreign_run(client, two_projects):
    p1, _p2, run2, _rag2 = two_projects
    resp = client.delete(f"/api/projects/{p1['id']}/runs/{run2['id']}")
    assert resp.status_code == 404
    assert db.get_run(run2["id"]) is not None, "foreign run must not be deleted"


def test_start_run_404s_for_foreign_run(client, two_projects):
    p1, _p2, run2, _rag2 = two_projects
    resp = client.post(f"/api/projects/{p1['id']}/runs/{run2['id']}/start")
    assert resp.status_code == 404


def test_stop_run_404s_for_foreign_run(client, two_projects):
    p1, _p2, run2, _rag2 = two_projects
    resp = client.post(f"/api/projects/{p1['id']}/runs/{run2['id']}/stop")
    assert resp.status_code == 404
    assert db.get_run(run2["id"])["status"] != "stopped"


def test_promote_run_404s_for_foreign_run(client, two_projects):
    p1, _p2, run2, _rag2 = two_projects
    resp = client.post(
        f"/api/projects/{p1['id']}/promote", json={"run_id": run2["id"]}
    )
    assert resp.status_code == 404
    assert db.get_project(p1["id"]).get("production_run") != run2["id"]


def test_merge_run_404s_for_foreign_run(client, two_projects):
    p1, _p2, run2, _rag2 = two_projects
    resp = client.post(f"/api/projects/{p1['id']}/runs/{run2['id']}/merge")
    assert resp.status_code == 404


def test_benchmark_run_404s_for_foreign_run(client, two_projects):
    p1, _p2, run2, _rag2 = two_projects
    resp = client.post(
        f"/api/projects/{p1['id']}/runs/{run2['id']}/benchmark", json={}
    )
    assert resp.status_code == 404


def test_update_rag_404s_for_foreign_rag(client, two_projects):
    p1, _p2, _run2, rag2 = two_projects
    resp = client.patch(
        f"/api/projects/{p1['id']}/rags/{rag2['id']}", json={"name": "hijacked"}
    )
    assert resp.status_code == 404
    assert db.get_rag(rag2["id"])["name"] == "rag-in-p2"


def test_delete_rag_404s_for_foreign_rag(client, two_projects):
    p1, _p2, _run2, rag2 = two_projects
    resp = client.delete(f"/api/projects/{p1['id']}/rags/{rag2['id']}")
    assert resp.status_code == 404
    assert db.get_rag(rag2["id"]) is not None


def test_query_rag_404s_for_foreign_rag(client, two_projects):
    p1, _p2, _run2, rag2 = two_projects
    resp = client.post(
        f"/api/projects/{p1['id']}/rags/{rag2['id']}/query", json={"query": "hi"}
    )
    assert resp.status_code == 404


def test_rag_stats_404s_for_foreign_rag(client, two_projects):
    p1, _p2, _run2, rag2 = two_projects
    resp = client.get(f"/api/projects/{p1['id']}/rags/{rag2['id']}/stats")
    assert resp.status_code == 404


def test_get_run_200s_for_own_project(client, two_projects):
    """The fix must not break the legitimate same-project case."""
    _p1, p2, run2, _rag2 = two_projects
    resp = client.get(f"/api/projects/{p2['id']}/runs/{run2['id']}")
    assert resp.status_code == 200
    assert resp.json()["id"] == run2["id"]


def test_update_project_404s_for_missing_project(client):
    resp = client.patch(f"/api/projects/{MISSING_ID}", json={"name": "x"})
    assert resp.status_code == 404


def test_create_rag_404s_for_missing_project(client):
    resp = client.post(f"/api/projects/{MISSING_ID}/rags", json={"name": "x"})
    assert resp.status_code == 404


def test_create_run_404s_for_missing_project(client):
    resp = client.post(f"/api/projects/{MISSING_ID}/runs", json={"name": "x"})
    assert resp.status_code == 404


# ── updates.py: ?full=1 query param must actually work ────────────────────


def test_update_status_full_param_returns_untruncated_log(client):
    row = db.create_update(mode="check", options={}, triggered_by="test")
    long_text = "x" * 5000
    db.append_update_log(row["id"], long_text)

    truncated = client.get(f"/api/system/update/{row['id']}")
    assert truncated.status_code == 200
    assert len(truncated.json()["log_tail"]) == 4000, (
        "default (no ?full) must stay truncated to the last 4000 chars"
    )

    full = client.get(f"/api/system/update/{row['id']}", params={"full": "1"})
    assert full.status_code == 200
    assert len(full.json()["log_tail"]) == 5000, (
        "?full=1 must return the untruncated log_text, per the route's own "
        "docstring — it used to be hardcoded False and unreachable"
    )


# ── data.py: upload filename must not escape data_dir ─────────────────────


def test_upload_sanitizes_path_traversal_filename(client, tmp_path, monkeypatch):
    # routes/data.py did ``from finetune_studio.config import settings`` at
    # import time, binding its own module-level name to that object. Patch
    # the attribute on that same object in place (not finetune_studio.config
    # .settings, which other fixtures may have already replaced wholesale)
    # so the route actually observes the override.
    from finetune_studio.webui.routes import data as data_route_module

    data_dir = tmp_path / "data_dir"
    data_dir.mkdir()
    monkeypatch.setattr(data_route_module.settings, "data_dir", str(data_dir))

    outside_marker = tmp_path / "escaped.txt"
    assert not outside_marker.exists()

    resp = client.post(
        "/api/data/upload",
        files={"file": ("../escaped.txt", b"payload", "text/plain")},
    )
    assert resp.status_code == 200
    body = resp.json()
    assert body["name"] == "escaped.txt", "filename must be reduced to its basename"
    assert not outside_marker.exists(), (
        "upload must never write outside settings.data_dir via a "
        "directory-traversal filename"
    )
    written = data_dir / "escaped.txt"
    assert written.is_file()
    assert written.read_bytes() == b"payload"
