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


# ── The 24 remaining project-scoped GET routes (measured 2026-09-26) ──
#
# A leak probe over the app's own route table (59 project-scoped GET routes)
# found these still answering 200 for a nonexistent project. Two shapes of
# defect: routes that never checked the pid at all, and routes that returned
# ``{"error": ...}`` as a normal dict, which FastAPI turns into a 200.

LEAKING_ROUTES = [
    # benchmarks.py
    "/api/benchmarks/projects/{pid}/benchmarks/{bid}/cases",
    "/api/benchmarks/projects/{pid}/runs",
    "/api/benchmarks/projects/{pid}/runs/{rid}/history",
    # projects.py
    "/api/projects/{pid}",
    "/api/projects/{pid}/data-prep/audit",
    "/api/projects/{pid}/data-prep/chat/tools",
    "/api/projects/{pid}/data-prep/export",
    "/api/projects/{pid}/data-prep/ingestion-log",
    "/api/projects/{pid}/data-prep/qa",
    "/api/projects/{pid}/data-prep/runs/{run_id}/events",
    "/api/projects/{pid}/data-prep/sources",
    "/api/projects/{pid}/exports",
    "/api/projects/{pid}/exports/{eid}/events",
    "/api/projects/{pid}/rag",
    "/api/projects/{pid}/rag/build/progress",
    "/api/projects/{pid}/rag/build/status",
    "/api/projects/{pid}/rags",
    "/api/projects/{pid}/rags/{rid}/stats",
    "/api/projects/{pid}/runs",
    "/api/projects/{pid}/runs/{rid}",
    "/api/projects/{pid}/runs/{rid}/benchmarks",
    "/api/projects/{pid}/runs/{rid}/exports",
    # testing.py
    "/api/testing/projects/{pid}/training-datasets",
    # training.py
    "/api/training/runs/{pid}",
]

SUBS = {
    "rid": "zz-no-such-run-zz",
    "bid": "zz-no-such-bench-zz",
    "eid": "zz-no-such-export-zz",
    "run_id": "zz-no-such-run-zz",
}


def _fill(template: str) -> str:
    """Substitute a bogus pid plus a bogus sub-resource id."""
    out = template
    for key, value in SUBS.items():
        out = out.replace("{" + key + "}", value)
    return out.replace("{pid}", MISSING_PID)


@pytest.mark.parametrize("template", LEAKING_ROUTES)
def test_missing_project_is_404(client: TestClient, template: str) -> None:
    """Every project-scoped GET route owes a caller a 404, never a 200.

    A 200 with an ``error`` key in the body reads as success to an XHR, so
    the caller only discovers the project was missing by sniffing the body.
    """
    url = _fill(template)
    resp = client.get(url)
    assert resp.status_code == 404, (
        f"{url} answered {resp.status_code} for a missing project; a caller "
        f"would treat that as success. body={resp.text[:200]}"
    )


@pytest.mark.parametrize(
    "template",
    [
        # SSE routes must reject before the stream opens, so no 200 with a
        # `data:` error frame can ever be emitted.
        "/api/projects/{pid}/data-prep/runs/{run_id}/events",
        "/api/projects/{pid}/exports/{eid}/events",
        "/api/projects/{pid}/rag/build/progress",
    ],
)
def test_streaming_routes_reject_before_the_stream(
    client: TestClient, template: str,
) -> None:
    """No SSE route may answer 200 and bury the failure in a data frame."""
    url = _fill(template)
    resp = client.get(url)
    assert resp.status_code == 404, (
        f"{url} answered {resp.status_code}; the stream started and only "
        f"reported the failure inside a frame. body={resp.text[:200]}"
    )
    assert not resp.text.startswith("data:"), (
        f"{url} emitted an SSE body instead of rejecting the bad project"
    )


def test_rag_status_does_not_leak_a_corpus_path(client: TestClient) -> None:
    """A bad pid must not get a real absolute ``corpus_dir`` back.

    ``/api/projects/{pid}/rag`` builds its path from the pid, so without a
    guard a nonexistent project got a 200 naming a directory on this host.
    """
    resp = client.get(f"/api/projects/{MISSING_PID}/rag")
    assert resp.status_code == 404
    assert "corpus_dir" not in resp.text, (
        f"a missing project leaked its corpus_dir: {resp.text[:200]}"
    )


@pytest.mark.parametrize(
    "template",
    [
        "/api/projects/{pid}/runs",
        "/api/projects/{pid}/rags",
        "/api/projects/{pid}/exports",
        "/api/projects/{pid}/data-prep/sources",
    ],
)
def test_a_real_project_still_answers_200(
    client: TestClient, template: str,
) -> None:
    """The guard must not turn a valid project into a 404 (regression gate).

    Each of these is a "list" route that returns an empty collection for a
    project that exists but has no children — that empty answer is what the
    UI relies on to render its empty state.
    """
    from finetune_studio import db

    project = db.create_project(name="valid-project-404-probe")
    try:
        pid = project["id"] if isinstance(project, dict) else project
        resp = client.get(_fill(template).replace(MISSING_PID, pid))
        assert resp.status_code == 200, (
            f"{template} answered {resp.status_code} for a real project; the "
            f"guard is too broad. body={resp.text[:200]}"
        )
    finally:
        db.delete_project(pid)


def test_a_real_project_with_no_runs_still_returns_an_empty_list(
    client: TestClient,
) -> None:
    """A brand-new project's run list is ``[]``, not a 404 or an error body."""
    from finetune_studio import db

    project = db.create_project(name="empty-runs-404-probe")
    try:
        pid = project["id"] if isinstance(project, dict) else project
        resp = client.get(f"/api/projects/{pid}/runs")
        assert resp.status_code == 200
        assert resp.json() == [], f"expected [], got {resp.text[:200]}"
    finally:
        db.delete_project(pid)
