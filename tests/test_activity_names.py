"""Activity rows name the request, not only its HTTP verb (2026-11-04).

The live feed held 19,057 rows that said only "Update" (Pairs page edits), 1,538 "Create" (chat) and so on: one row per
mutating request, named by verb. ``webui.activity_names`` names each route the app's pages call; anything else falls
back to its resource words. These tests pin the table to the real routes and the live paths to real names.
"""
from __future__ import annotations

import re

import pytest

from finetune_studio.webui.activity_names import NAMES, describe
from finetune_studio.webui.app import _activity_summary, app

_BARE = {"Create", "Update", "Delete", "Operation"}


def _shape(route: str) -> str:
    return re.sub(r"\{[^}]*\}|\*", "{}", route)


def test_every_named_route_is_a_real_route() -> None:
    # Routers are mounted lazily; the OpenAPI document lists every API route with its methods.
    real = {(m.upper(), _shape(p)) for p, ops in app.openapi()["paths"].items() for m in ops}
    missing = [(m, route) for (m, route) in NAMES if (m, _shape(route)) not in real]
    assert missing == []


# Paths as the live activity log recorded them (ids from a real project), the most frequent bare-verb rows first.
LIVE = [
    ("PATCH", "/api/projects/959fc605/data-prep/qa/3f2a9c1b7d4e", "Edit a Q&A pair"),
    ("POST", "/api/projects/959fc605/data-prep/qa", "Add a Q&A pair"),
    ("POST", "/api/inference/chat", "Chat with the loaded model"),
    ("POST", "/api/projects/959fc605/brain/chat", "Smart Brain chat"),
    ("POST", "/api/projects/959fc605/brain/search", "Smart Brain search"),
    ("DELETE", "/api/projects/959fc605", "Delete project"),
    ("POST", "/api/guide/chat", "Ask the Guide"),
    ("POST", "/api/testing/projects/959fc605/runs/248bddcc/judge", "Judge a test run"),
    ("POST", "/api/providers/local-default/load", "Load model"),
    ("PUT", "/api/testing/projects/959fc605/runs/248bddcc/cases/c3fb8380/verdict", "Set a verdict"),
    ("POST", "/api/projects/959fc605/files/c79a2dff-b560d0ea499ba53c/purge", "Delete a file from the trash"),
    ("PUT", "/api/settings/helper", "Save helper settings"),
    ("PUT", "/api/settings/assistant/jobs/mining", "Choose the assistant for a job"),
    ("POST", "/api/settings/vision/test", "Test the vision connection"),
    ("POST", "/api/inference/load", "Load model"),
    ("POST", "/api/auth/login", "Sign in"),
]


@pytest.mark.parametrize(("method", "path", "name"), LIVE)
def test_live_requests_get_a_name(method: str, path: str, name: str) -> None:
    assert _activity_summary(path, method) == name


def test_an_unlisted_route_is_named_by_its_resource_words_not_its_verb() -> None:
    assert describe("/api/projects/959fc605/usecases/abc12345/evaluate", "POST") == "Create: usecases › evaluate"
    assert describe("/api/something-new", "DELETE") == "Delete: something-new"


def test_no_named_route_is_shadowed_by_the_older_labels() -> None:
    """The specific labels in _activity_summary run first; none may swallow a table entry with a different name."""
    for (method, route), name in NAMES.items():
        path = route.replace("{}", "959fc605").replace("*", "local-default")
        got = _activity_summary(path, method)
        assert got not in _BARE, (method, route, got)
        if got != name:  # an older, more specific label won: it must still name the thing
            assert got in {"Upload + parse files", "Load model", "Unload model", "Export model", "Run test suite"}, \
                (method, route, got)


@pytest.mark.parametrize(("path", "pid"), [
    ("/api/projects/959fc605/data-prep/qa/3f2a9c1b7d4e", "959fc605"),
    ("/api/testing/projects/959fc605/runs/248bddcc/cases/c3fb8380/verdict", "959fc605"),
    ("/api/compare/projects/959fc605/groups/g1", "959fc605"),
    ("/api/benchmarks/projects/959fc605/suites/import", "959fc605"),
    ("/api/chat-v2/projects/959fc605/chat", "959fc605"),
    ("/api/projects", ""),
    ("/api/inference/chat", ""),
])
def test_activity_rows_sit_under_the_project_the_request_acts_on(path: str, pid: str) -> None:
    from finetune_studio.webui.app import _activity_project_id

    assert _activity_project_id(path) == pid


def test_the_delete_project_row_is_visible_and_names_the_project(client) -> None:
    """The row of a project delete used to sit under the deleted project, so the feed (which hides rows of missing
    projects) never showed it, and the project's other rows stayed in the table forever."""
    from finetune_studio import db

    pid = client.post("/api/projects", json={"name": "Short-lived", "base_model": "x/test"}).json()["id"]
    assert client.post(f"/api/projects/{pid}/versions", json={}).status_code < 500   # leaves a row under it
    assert client.delete(f"/api/projects/{pid}").status_code == 200
    rows = [e for e in db.list_activity_events_recent(50) if pid in (e["path"] or "")]
    assert [(e["project_id"], e["message"]) for e in rows] == [("", "Delete project · Short-lived")]


def test_requests_naming_their_project_in_the_body_sit_under_it(client) -> None:
    """Create project, Start training run and the testing starts carry the project in the body (or create it):
    their Activity rows used to say "system"."""
    from finetune_studio import db

    pid = client.post("/api/projects", json={"name": "Body-named", "base_model": "x/test"}).json()["id"]
    client.post("/api/training/start", json={"project_id": pid})            # refused (no data): still its row
    client.post("/api/testing/run-suite", json={"project_id": pid})         # refused (no suite): still its row
    rows = {e["path"]: e["project_id"] for e in db.list_activity_events_recent(20)}
    assert rows["/api/projects"] == pid
    assert rows["/api/training/start"] == pid
    assert rows["/api/testing/run-suite"] == pid
