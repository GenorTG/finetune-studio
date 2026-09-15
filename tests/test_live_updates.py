"""Regression: live SSE endpoints + client wiring for long-running status."""

from __future__ import annotations

import asyncio
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
_APP_JS = _ROOT / "src" / "finetune_studio" / "webui" / "static" / "js" / "app.js"
_ACTIVITY_JS = _ROOT / "src" / "finetune_studio" / "webui" / "static" / "js" / "activity.js"
_TRAINING_JS = _ROOT / "src" / "finetune_studio" / "webui" / "static" / "js" / "training.js"
_TRAINING_HTML = (
    _ROOT / "src" / "finetune_studio" / "webui" / "templates" / "project_training.html"
)
_TESTING_HTML = (
    _ROOT / "src" / "finetune_studio" / "webui" / "templates" / "project_testing.html"
)
_EXPORT_HTML = (
    _ROOT / "src" / "finetune_studio" / "webui" / "templates" / "export_models.html"
)
_BASE_HTML = _ROOT / "src" / "finetune_studio" / "webui" / "templates" / "base.html"


def test_app_js_exposes_subscribe_helper() -> None:
    src = _APP_JS.read_text(encoding="utf-8")
    assert "function subscribe(" in src
    assert "EventSource" in src
    assert "pollUrl" in src
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


def test_export_page_wires_export_events() -> None:
    html = _EXPORT_HTML.read_text(encoding="utf-8")
    assert "/exports/" in html and "/events" in html
    assert "export-result" in html
    assert 'value="awq"' not in html


def test_sse_routes_registered(client) -> None:
    """One-shot fallbacks stay available; OpenAPI lists the SSE endpoints."""
    assert client.get("/api/activity").status_code == 200
    assert client.get("/api/training/status").status_code == 200
    assert client.get("/api/testing/status").status_code == 200
    paths = set(client.app.openapi().get("paths", {}))
    assert "/api/activity/events" in paths
    assert "/api/training/progress" in paths
    assert "/api/testing/events" in paths
    assert "/api/projects/{pid}/exports/{eid}/events" in paths


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


def test_export_events_unknown_export(client) -> None:
    pid = client.post("/api/projects", json={"name": "Export SSE"}).json()["id"]
    # Terminal failure frame for unknown export — generator exits, so TestClient
    # can consume the full body without hanging.
    r = client.get(f"/api/projects/{pid}/exports/does-not-exist/events")
    assert r.status_code == 200
    assert "text/event-stream" in (r.headers.get("content-type") or "")
    assert b"not found" in r.content or b"failed" in r.content
