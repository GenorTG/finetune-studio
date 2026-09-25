"""A nonexistent project must be a 404, never a 200 with an error body.

Found 2026-09-25 during the CPU-only WebUI audit. Two handlers returned
``{"error": "Project not found"}`` as a normal dict, so FastAPI answered
**200** and the XHR "succeeded" — the caller had to sniff the body for an
``error`` key to discover the project was missing. Every other project route
in this app uses ``HTTPException(404, ...)`` or ``_project_or_404``.

A third defect lived next door: ``compare_runs`` validated its query params
*before* the project, so a bad ``pid`` was reported as the unrelated
"run_a and run_b required" (400) and the caller hunted for a missing query
param instead of the project that did not exist.

These are status-code regressions: they assert the wire contract, not
internal structure, so a refactor cannot quietly invert them.
"""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from finetune_studio.webui.app import app

MISSING_PID = "deadbeef"
"""A syntactically valid project id that cannot exist (8 hex chars)."""


@pytest.fixture(scope="module")
def client() -> TestClient:
    return TestClient(app, raise_server_exceptions=False)


@pytest.mark.parametrize(
    "url",
    [
        f"/api/chat-v2/projects/{MISSING_PID}/context",
        f"/api/chat-v2/projects/{MISSING_PID}/chat",
    ],
)
def test_chat_v2_missing_project_is_404(client: TestClient, url: str) -> None:
    """The two chat_v2 sites that used to answer 200 with an error body."""
    resp = client.get(url) if url.endswith("context") else client.post(
        url, json={"messages": [{"role": "user", "content": "hi"}]},
    )
    assert resp.status_code == 404, (
        f"{url} answered {resp.status_code} for a missing project; a caller "
        f"would treat that as success. body={resp.text[:200]}"
    )


def test_context_response_model_does_not_swallow_the_404(client: TestClient) -> None:
    """A 200 body must never be the only signal that a project is missing."""
    resp = client.get(f"/api/chat-v2/projects/{MISSING_PID}/context")
    assert resp.status_code != 200, "missing project leaked through as a 200"
    assert "error" not in resp.json() or resp.status_code >= 400


@pytest.mark.parametrize(
    ("query", "label"),
    [
        ("", "no run params"),
        ("?run_a=aaa", "only run_a"),
        ("?run_b=bbb", "only run_b"),
    ],
)
def test_compare_reports_the_project_before_the_params(
    client: TestClient, query: str, label: str,
) -> None:
    """A missing project outranks missing run params (regression: was 400)."""
    resp = client.get(f"/api/benchmarks/projects/{MISSING_PID}/compare{query}")
    assert resp.status_code == 404, (
        f"compare with {label} answered {resp.status_code} for a missing "
        f"project; expected 404. body={resp.text[:200]}"
    )
    assert "run_a and run_b" not in resp.text, (
        "compare blamed the run params for what is actually a missing project"
    )


def test_compare_still_reports_missing_runs_as_400_when_project_exists(
    client: TestClient,
) -> None:
    """The fix must not swallow the genuine run-param error.

    A 404 is only correct for the *project*. When the project exists but the
    runs are absent, 400 'run_a and run_b required' is the right answer, so
    this pins that the new guard did not reorder the cases into a 404-everything.
    """
    from finetune_studio import db

    project = db.create_project(name="compare-404-probe")
    try:
        pid = project["id"] if isinstance(project, dict) else project
        resp = client.get(f"/api/benchmarks/projects/{pid}/compare")
        assert resp.status_code == 400, (
            "with a real project and no run params, compare must still be 400; "
            f"got {resp.status_code}. body={resp.text[:200]}"
        )
        assert "run_a and run_b required" in resp.text
    finally:
        db.delete_project(pid)
