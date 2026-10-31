"""Regression: live SSE endpoints + client wiring for long-running status."""

from __future__ import annotations

import asyncio
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
_APP_JS = _ROOT / "src" / "finetune_studio" / "webui" / "static" / "js" / "app.js"
_ACTIVITY_JS = _ROOT / "src" / "finetune_studio" / "webui" / "static" / "js" / "activity.js"
_SETTINGS_JS = _ROOT / "src" / "finetune_studio" / "webui" / "static" / "js" / "settings.js"
_TRAINING_HTML = (
    _ROOT / "src" / "finetune_studio" / "webui" / "templates" / "project_training.html"
)
_TESTING_HTML = (
    _ROOT / "src" / "finetune_studio" / "webui" / "templates" / "project_testing.html"
)
_EXPORT_HTML = (
    _ROOT / "src" / "finetune_studio" / "webui" / "templates" / "export_models.html"
)
_RAG_HTML = _ROOT / "src" / "finetune_studio" / "webui" / "templates" / "brain.html"
_DATA_PREP_HTML = (
    _ROOT / "src" / "finetune_studio" / "webui" / "templates" / "data_prep.html"
)
_BASE_HTML = _ROOT / "src" / "finetune_studio" / "webui" / "templates" / "base.html"


def test_app_js_exposes_subscribe_helper() -> None:
    src = _APP_JS.read_text(encoding="utf-8")
    assert "function subscribe(" in src
    assert "EventSource" in src
    assert "pollUrl" in src
    assert "never a 2s full-panel redraw" in src
    fts_block = src.split("window.fts =", 1)[1].split("};", 1)[0]
    assert "subscribe" in fts_block


def test_activity_uses_sse_not_2s_poll() -> None:
    src = _ACTIVITY_JS.read_text(encoding="utf-8")
    assert "/api/activity/events" in src
    assert "subscribe" in src
    assert "setInterval(refresh, 2000)" not in src
    assert "every 2s" not in src.lower()
    base = _BASE_HTML.read_text(encoding="utf-8")
    assert "Updated every 2s" not in base
    assert "Live</span>" in base


def test_training_monitor_uses_progress_sse() -> None:
    html = _TRAINING_HTML.read_text(encoding="utf-8")
    assert "/api/training/progress" in html
    assert "fts.subscribe" in html or "window.fts && window.fts.subscribe" in html
    assert "setInterval(tick, 2000)" not in html
    assert "Updated every 2s" not in html
    assert "Streaming while training" in html


def test_testing_page_uses_testing_events() -> None:
    html = _TESTING_HTML.read_text(encoding="utf-8")
    assert "/api/testing/events" in html
    assert "setInterval(tickLive, 2000)" not in html
    assert "fts.subscribe" in html or "window.fts.subscribe" in html
    assert "rv-export-json" in html


def test_export_page_wires_export_events() -> None:
    html = _EXPORT_HTML.read_text(encoding="utf-8")
    assert "/exports/" in html and "/events" in html
    assert "export-result" in html
    assert 'value="awq"' not in html


def test_brain_build_uses_subscribe_not_1_5s_poll() -> None:
    html = _RAG_HTML.read_text(encoding="utf-8")
    assert "/brain/build/progress" in html
    assert "fts.subscribe" in html or "window.fts && window.fts.subscribe" in html
    assert "fallbackMs: 5000" in html
    assert "/brain/build/status" in html
    assert "1500" not in html
    assert "loadBrainStatus" in html
    assert "runQuery" in html
    assert "run-kv-table" in html
    assert "Debug JSON" in html
    # A build is a job: the page follows its build row to the end (and re-attaches on load).
    assert "/brain/builds/" in html and "followBuild" in html and "resumeBuild" in html
    assert "docs, " in html and "chunks" in html and " indexed." in html


def test_brain_build_answers_202_then_the_build_row_carries_the_result(client, tmp_path, monkeypatch) -> None:
    """POST /brain/build starts a job (a web restart does not end it): 202 + build_id, building=true; the build row
    then carries the counts."""
    from pathlib import Path
    from unittest.mock import MagicMock, patch

    from finetune_studio.data.fs.paths import brain_index_dir
    from tests.knowledge_helpers import add_source, wait_build

    pid = client.post("/api/projects", json={"name": "RAG sync status"}).json()["id"]
    add_source(pid, "a.txt", "hi there, a sentence to index")
    corpus = brain_index_dir(pid)
    mock_rag = MagicMock()
    mock_rag.build_from_documents.return_value = {
        "documents": 1,
        "chunks": 2,
        "vector_dim": 8,
        "skipped": 0,
    }
    with patch(
        "finetune_studio.smart_brain.BrainIndex",
        return_value=mock_rag,
    ):
        r = client.post(f"/api/projects/{pid}/brain/build", json={"reset": False})
        build = wait_build(client, pid, r)
    body = r.json()
    assert body["ok"] is True
    assert body["building"] is True and body["build_id"] == build["build_id"]
    assert Path(body["corpus_dir"]) == corpus
    assert build["ok"] is True and build["status"] == "done"
    assert build["documents"] == 1
    assert build["chunks"] == 2
    assert build["vector_dim"] == 8


def test_data_prep_polls_mining_jobs_only_while_they_run() -> None:
    html = _DATA_PREP_HTML.read_text(encoding="utf-8")
    assert "/jobs?active=true" in html
    # live 2 s refresh only while a job is queued or running, idle otherwise
    assert "if (Object.keys(_dpJobs).length) _dpFilesTimer = setTimeout(dpLoadFiles, 2000)" in html
    assert "dp-results-table" in html
    # Raw-JSON debug box removed from data prep (plain-language audit 2026-10-03).
    assert "Debug JSON" not in html
    assert "setInterval(prepRefreshResults, 2000)" not in html


def test_settings_update_uses_sse_not_2s_poll() -> None:
    src = _SETTINGS_JS.read_text(encoding="utf-8")
    assert "/api/system/update/events" in src
    assert "fts.subscribe" in src or "window.fts && window.fts.subscribe" in src
    assert "fallbackMs: 5000" in src
    assert "setInterval(() => updTick(false), 2000)" not in src


def test_sse_routes_registered(client) -> None:
    """One-shot fallbacks stay available; OpenAPI lists the SSE endpoints."""
    assert client.get("/api/activity").status_code == 200
    assert client.get("/api/training/status").status_code == 200
    assert client.get("/api/testing/status").status_code == 200
    assert client.get("/api/system/update/latest").status_code == 200
    paths = set(client.app.openapi().get("paths", {}))
    assert "/api/activity/events" in paths
    assert "/api/training/progress" in paths
    assert "/api/testing/events" in paths
    assert "/api/projects/{pid}/exports/{eid}/events" in paths
    assert "/api/system/update/events" in paths
    assert "/api/projects/{pid}/brain/build/progress" in paths
    assert "/api/projects/{pid}/brain/build/status" in paths
    assert "/api/projects/{pid}/mining/jobs" in paths


def test_training_events_emits_snapshot() -> None:
    from finetune_studio.training.monitor import training_events, training_snapshot
    from finetune_studio.webui.app import training_engine

    snap = training_snapshot(training_engine)
    assert "status" in snap
    assert "log_lines" in snap

    async def _first_frame() -> str:
        agen = training_events(training_engine)
        try:
            return await agen.__anext__()
        finally:
            await agen.aclose()

    frame = asyncio.run(_first_frame())
    assert frame.startswith("data: ")
    assert "status" in frame


def test_activity_collect_snapshot() -> None:
    from finetune_studio.webui.routes.activity import collect_activity

    payload = collect_activity()
    assert "tasks" in payload
    assert "active_count" in payload
    assert isinstance(payload["tasks"], list)


def test_activity_collect_handles_none_started_at(monkeypatch) -> None:
    """Regression: explicit started_at=None must not break sort (unary -)."""
    from finetune_studio.webui import download_jobs
    from finetune_studio.webui.routes.activity import collect_activity

    # A running download (its job process alive) as the activity feed reads it.
    monkeypatch.setattr(download_jobs, "live_downloads", lambda: [{
        "job_id": "job-none-started",
        "status": "downloading",
        "repo_id": "org/model-with-none-started",
        "started_at": None,
        "bytes_done": 0,
        "bytes_total": 0,
    }])
    payload = collect_activity()
    assert "tasks" in payload
    downloads = [t for t in payload["tasks"] if t.get("kind") == "download"]
    assert any(t.get("id") == "job-none-started" for t in downloads)


def test_export_events_unknown_export(client) -> None:
    pid = client.post("/api/projects", json={"name": "Export SSE"}).json()["id"]
    # Terminal failure frame for unknown export — generator exits, so TestClient
    # can consume the full body without hanging.
    r = client.get(f"/api/projects/{pid}/exports/does-not-exist/events")
    assert r.status_code == 200
    assert "text/event-stream" in (r.headers.get("content-type") or "")
    assert b"not found" in r.content or b"failed" in r.content


def test_update_events_idle_exits(client) -> None:
    r = client.get("/api/system/update/events")
    assert r.status_code == 200
    assert "text/event-stream" in (r.headers.get("content-type") or "")
    assert b"exists" in r.content


def test_brain_build_status_snapshot(client) -> None:
    pid = client.post("/api/projects", json={"name": "RAG status"}).json()["id"]
    r = client.get(f"/api/projects/{pid}/brain/build/status")
    assert r.status_code == 200
    body = r.json()
    assert "phase" in body
    assert "files_done" in body
    assert "files_total" in body


def test_mining_job_cancel_unknown_is_404(client) -> None:
    pid = client.post("/api/projects", json={"name": "Prep status"}).json()["id"]
    r = client.post(f"/api/projects/{pid}/mining/jobs/does-not-exist/cancel")
    assert r.status_code == 404 and r.json()["detail"] == "unknown job"


def test_brain_source_clear_resets_stale_retrieval_and_chat_feedback() -> None:
    html = _RAG_HTML.read_text(encoding="utf-8")
    assert "function clearBrainTransientOutput()" in html
    helper = html.split("function clearBrainTransientOutput()", 1)[1].split("\n  }", 1)[0]
    for element_id in ("q-status", "q-results", "chat-messages", "build-status", "quick-index-status"):
        assert f"getElementById('{element_id}')" in helper

    for action in ("clearAllDocuments", "wipeIndex"):
        prefix = f"window.{action} = async function() {{"
        assert prefix in html
        handler = html.split(prefix, 1)[1].split("\n  };", 1)[0]
        assert "clearBrainTransientOutput()" in handler
        assert "loadBrainStatus()" in handler
        assert "loadIndexedDocs()" in handler
