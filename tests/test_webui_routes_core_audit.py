"""Regression tests for the B1 (webui/routes core) documentation audit.

Pins fixes made 2026-10-01 (a third, the ``routes/data.py`` upload traversal, went away with the legacy
``/api/data/*`` routes on 2026-10-25):

1. ``projects.py`` run sub-resource routes used to trust the ``{rid}``
   path param without checking it belongs to the ``{pid}`` in the URL — a
   run id from a different project was fully readable and mutable through
   any other project's URL. Every such route now 404s for a foreign-owned or
   missing resource (see ``_get_owned_run`` in ``routes/projects.py``).
2. ``routes/updates.py::get_update_status`` documented a ``?full=1`` query
   param in its docstring but never actually declared it on the route, so
   ``full`` was hardcoded ``False`` and the full log text was unreachable.
"""

from __future__ import annotations

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
    """Two real projects, the second with a run, for cross-project checks."""
    p1 = db.create_project(name="audit-b1-proj-1")
    p2 = db.create_project(name="audit-b1-proj-2")
    run2 = db.create_run(project_id=p2["id"], name="run-in-p2")
    return p1, p2, run2


# ── projects.py: cross-project run ownership ──────────────────────────


def test_get_run_404s_for_run_owned_by_another_project(client, two_projects):
    p1, p2, run2 = two_projects
    resp = client.get(f"/api/projects/{p1['id']}/runs/{run2['id']}")
    assert resp.status_code == 404, (
        f"run {run2['id']} belongs to {p2['id']}, not {p1['id']}; expected 404, "
        f"got {resp.status_code}: {resp.text[:200]}"
    )


def test_update_run_404s_for_foreign_run(client, two_projects):
    p1, _p2, run2 = two_projects
    resp = client.patch(
        f"/api/projects/{p1['id']}/runs/{run2['id']}", json={"name": "hijacked"}
    )
    assert resp.status_code == 404
    # The run itself must be untouched.
    fresh = db.get_run(run2["id"])
    assert fresh["name"] == "run-in-p2"


def test_delete_run_404s_for_foreign_run(client, two_projects):
    p1, _p2, run2 = two_projects
    resp = client.delete(f"/api/projects/{p1['id']}/runs/{run2['id']}")
    assert resp.status_code == 404
    assert db.get_run(run2["id"]) is not None, "foreign run must not be deleted"


def test_promote_run_404s_for_foreign_run(client, two_projects):
    p1, _p2, run2 = two_projects
    resp = client.post(
        f"/api/projects/{p1['id']}/promote", json={"run_id": run2["id"]}
    )
    assert resp.status_code == 404
    assert db.get_project(p1["id"]).get("production_run") != run2["id"]


def test_get_run_200s_for_own_project(client, two_projects):
    """The fix must not break the legitimate same-project case."""
    _p1, p2, run2 = two_projects
    resp = client.get(f"/api/projects/{p2['id']}/runs/{run2['id']}")
    assert resp.status_code == 200
    assert resp.json()["id"] == run2["id"]


def test_update_project_404s_for_missing_project(client):
    resp = client.patch(f"/api/projects/{MISSING_ID}", json={"name": "x"})
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


