"""Regression: live SSE endpoints + client wiring for long-running status."""

from __future__ import annotations

import asyncio
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
_APP_JS = _ROOT / "src" / "finetune_studio" / "webui" / "static" / "js" / "app.js"
_ACTIVITY_JS = _ROOT / "src" / "finetune_studio" / "webui" / "static" / "js" / "activity.js"
_TRAINING_JS = _ROOT / "src" / "finetune_studio" / "webui" / "static" / "js" / "training.js"
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
_RAG_HTML = _ROOT / "src" / "finetune_studio" / "webui" / "templates" / "rag.html"
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
    js = _TRAINING_JS.read_text(encoding="utf-8")
    assert "/api/training/progress" in js
    assert "setInterval(pollStatus, 2000)" not in js


def test_testing_page_uses_testing_events() -> None:
    html = _TESTING_HTML.read_text(encoding="utf-8")
    assert "/api/testing/events" in html
    assert "setInterval(tickLive, 2000)" not in html
    assert "t-results-debug" in html
    assert "Debug JSON" in html


def test_export_page_wires_export_events() -> None:
    html = _EXPORT_HTML.read_text(encoding="utf-8")
    assert "/exports/" in html and "/events" in html
    assert "export-result" in html
    assert 'value="awq"' not in html


def test_rag_build_uses_subscribe_not_1_5s_poll() -> None:
    html = _RAG_HTML.read_text(encoding="utf-8")
    assert "/rag/build/progress" in html
    assert "fts.subscribe" in html or "window.fts && window.fts.subscribe" in html
    assert "fallbackMs: 5000" in html
    assert "/rag/build/status" in html
    assert "1500" not in html
    assert "loadRagStatus" in html
    assert "runQuery" in html
    assert "run-kv-table" in html
    assert "Debug JSON" in html
    # Sync build path: UI must finish when building===false (no false "Queued…").
    assert "!r.building" in html or "r.building === false" in html or "if (!r.building)" in html
    assert "docs, " in html and "chunks indexed" in html


def test_rag_build_sync_response_not_building(client, tmp_path, monkeypatch) -> None:
    """POST /rag/build is synchronous — response must not say building=true."""
    from pathlib import Path
    from unittest.mock import MagicMock, patch

    pid = client.post("/api/projects", json={"name": "RAG sync status"}).json()["id"]
    corpus = tmp_path / "rag_corpora" / pid
    home = tmp_path / "home"
    files = home / ".finetune-studio" / "projects" / pid / "files"
    files.mkdir(parents=True)
    (files / "a.txt").write_text("hi", encoding="utf-8")
    monkeypatch.setattr(
        "finetune_studio.webui.routes.rag._corpus_dir",
        lambda p: corpus if p == pid else tmp_path / "x" / p,
    )
    monkeypatch.setattr(
        "finetune_studio.webui.routes.rag.Path.home",
        staticmethod(lambda: home),
    )
    mock_rag = MagicMock()
    mock_rag.build_from_directory.return_value = {
        "documents": 1,
        "chunks": 2,
        "vector_dim": 8,
        "skipped": 0,
    }
    with patch(
        "finetune_studio.data.rag_portable.PortableRAG",
        return_value=mock_rag,
    ):
        r = client.post(f"/api/projects/{pid}/rag/build", json={"reset": False})
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["ok"] is True
    assert body["building"] is False
    assert body["documents"] == 1
    assert body["chunks"] == 2
    assert Path(body["corpus_dir"]) == corpus


def test_data_prep_uses_subscribe_with_status_fallback() -> None:
    html = _DATA_PREP_HTML.read_text(encoding="utf-8")
    assert "/data-prep/runs/" in html and "/events" in html
    assert "fts.subscribe" in html or "window.fts && window.fts.subscribe" in html
    assert "fallbackMs: 5000" in html
    assert "dp-results-table" in html
    assert "Debug JSON" in html
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
    assert "/api/projects/{pid}/rag/build/progress" in paths
    assert "/api/projects/{pid}/rag/build/status" in paths
    assert "/api/projects/{pid}/data-prep/runs/{run_id}/events" in paths
    assert "/api/projects/{pid}/data-prep/runs/{run_id}" in paths


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
    from finetune_studio.webui.routes import hf_models
    from finetune_studio.webui.routes.activity import collect_activity

    monkeypatch.setitem(
        hf_models._DOWNLOADS,
        "job-none-started",
        {
            "status": "downloading",
            "repo_id": "org/model-with-none-started",
            "started_at": None,
            "bytes_done": 0,
            "bytes_total": 0,
        },
    )
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


def test_rag_build_status_snapshot(client) -> None:
    pid = client.post("/api/projects", json={"name": "RAG status"}).json()["id"]
    r = client.get(f"/api/projects/{pid}/rag/build/status")
    assert r.status_code == 200
    body = r.json()
    assert "phase" in body
    assert "files_done" in body
    assert "files_total" in body


def test_data_prep_run_status_unknown(client) -> None:
    pid = client.post("/api/projects", json={"name": "Prep status"}).json()["id"]
    r = client.get(f"/api/projects/{pid}/data-prep/runs/does-not-exist")
    assert r.status_code == 404
    assert r.json().get("stage") == "error"
