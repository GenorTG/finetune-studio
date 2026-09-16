"""Production UI reliability: nav readability, tables, upload refresh, loader."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock

import pytest
from fastapi.testclient import TestClient

_ROOT = Path(__file__).resolve().parents[1]
_CSS = _ROOT / "src" / "finetune_studio" / "webui" / "static" / "css" / "app.css"
_BASE = _ROOT / "src" / "finetune_studio" / "webui" / "templates" / "base.html"
_DATA_PREP = (
    _ROOT / "src" / "finetune_studio" / "webui" / "templates" / "data_prep.html"
)
_INFERENCE = (
    _ROOT / "src" / "finetune_studio" / "webui" / "templates" / "inference.html"
)
_RAG = _ROOT / "src" / "finetune_studio" / "webui" / "templates" / "rag.html"
_PROJECT_DATA = (
    _ROOT / "src" / "finetune_studio" / "webui" / "templates" / "project_data.html"
)


def _project(client: TestClient) -> str:
    r = client.post("/api/projects", json={"name": "UI Rel", "base_model": "x/t"})
    assert r.status_code in (200, 201), r.text
    return r.json()["id"]


def test_header_nav_css_readable_floor() -> None:
    """Session-bar / workspace nav must stay above the old tiny 10–12px floor."""
    css = _CSS.read_text(encoding="utf-8")
    assert ".sb-tabs .sb-tab" in css
    assert "font-size: 14px" in css
    assert ".workspace-subnav" in css
    assert "font-size: 13.5px" in css
    assert ".ws-label" in css
    # Mid-width media query must not collapse labels to ≤11px again.
    assert ".sb-tab-label { font-size: 13px; }" in css
    assert ".sb-tab-label { font-size: 11px; }" not in css
    assert ".sb-tab-label { font-size: 10px; }" not in css


def test_base_workspace_label_and_aria_current() -> None:
    base = _BASE.read_text(encoding="utf-8")
    assert 'class="ws-label"' in base
    assert "Workspace" in base
    assert 'aria-current="page"' in base


def test_table_fixed_layout_and_empty_colspan() -> None:
    css = _CSS.read_text(encoding="utf-8")
    assert "table-layout: fixed" in css
    assert "text-overflow: ellipsis" in css
    assert ".table td[colspan]" in css or ".empty-row td" in css

    dp = _DATA_PREP.read_text(encoding="utf-8")
    assert "table-layout: fixed" in dp
    assert 'class="fl-col-name"' in dp
    assert 'colspan="7"' in dp
    assert "empty-row" in dp

    pdata = _PROJECT_DATA.read_text(encoding="utf-8")
    assert 'colspan="8"' in pdata
    assert "empty-row" in pdata
    assert "cell-wrap" in pdata


def test_upload_refresh_paints_before_prefetch() -> None:
    """Successful upload must refresh the list in-place without waiting on conversions."""
    src = _DATA_PREP.read_text(encoding="utf-8")
    load_js = src.split("async function flLoadFiles", 1)[1].split(
        "async function flLoadTrash", 1
    )[0]
    assert "flRenderFiles()" in load_js
    assert "flPrefetchConversions" in load_js
    # Render must happen before awaiting prefetch (prefetch is fire-and-then).
    assert "flRenderFiles();\n    flPrefetchConversions" in load_js.replace(
        "\r\n", "\n"
    ) or "flRenderFiles();\r\n    flPrefetchConversions" in load_js

    upload_js = src.split("async function flUploadFiles", 1)[1].split(
        "function flShowUploadReport", 1
    )[0]
    assert "_flCurrentFolder = null" in upload_js
    assert "await flInit()" in upload_js
    assert "flShowUploadReport(data)" in upload_js
    assert "ALL FILES" in upload_js


def test_save_settings_defined_on_rag_page() -> None:
    src = _RAG.read_text(encoding="utf-8")
    assert 'onclick="saveSettings()"' in src
    assert "window.saveSettings" in src
    assert "/rag/settings" in src


def test_inference_ui_requires_loaded_confirmation() -> None:
    src = _INFERENCE.read_text(encoding="utf-8")
    assert "r.loaded === false" in src or "r.status === 'error'" in src
    assert "/api/inference/status" in src
    assert "still unloaded" in src or "still shows unloaded" in src


def test_load_missing_path_returns_explicit_failure(client: TestClient) -> None:
    r = client.post("/api/models/load", json={})
    assert r.status_code == 200, r.text
    data = r.json()
    assert data.get("status") == "error"
    assert data.get("loaded") is False
    assert "error" in data
    assert data.get("model") is None


def test_load_exception_returns_failure_not_loaded(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    engine = MagicMock()
    engine.model = None
    engine.vision = False

    def _boom(*_a, **_k):
        raise RuntimeError("CUDA OOM: tried to allocate 12 GiB")

    engine.load.side_effect = _boom
    monkeypatch.setattr("finetune_studio.webui.app.inference_engine", engine)

    r = client.post("/api/models/load", json={"path": "/tmp/missing-model.bin"})
    assert r.status_code == 200, r.text
    data = r.json()
    assert data["status"] == "error"
    assert data["loaded"] is False
    assert "CUDA OOM" in data["error"]
    assert data.get("model") is None


def test_load_success_requires_model_object(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    """If load() returns without holding a model, never claim status=loaded."""
    engine = MagicMock()
    engine.model = None
    engine.vision = False
    engine.load = MagicMock()
    monkeypatch.setattr("finetune_studio.webui.app.inference_engine", engine)

    r = client.post("/api/models/load", json={"path": "/models/fake"})
    assert r.status_code == 200, r.text
    data = r.json()
    assert data["status"] == "error"
    assert data["loaded"] is False
    assert "error" in data


def test_load_success_payload_when_model_held(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    engine = MagicMock()
    engine.model = object()
    engine.vision = False
    engine.load = MagicMock()
    monkeypatch.setattr("finetune_studio.webui.app.inference_engine", engine)

    r = client.post("/api/models/load", json={"path": "/models/ok"})
    assert r.status_code == 200, r.text
    data = r.json()
    assert data["status"] == "loaded"
    assert data["loaded"] is True
    assert data["model"] == "/models/ok"


def test_data_prep_page_has_upload_refresh_hooks(client: TestClient) -> None:
    pid = _project(client)
    r = client.get(f"/projects/{pid}/data-prep")
    assert r.status_code == 200, r.text
    body = r.text
    assert "flUploadFiles" in body
    assert "await flInit()" in body
    assert 'id="fl-files-body"' in body
    assert 'colspan="7"' in body


def test_responsive_header_gutter_and_780_breakpoint() -> None:
    """Header/tabs must shrink the absolute .sb-right reserve at mid widths."""
    css = _CSS.read_text(encoding="utf-8")
    assert "@media (max-width: 780px)" in css
    assert "@media (max-width: 900px)" in css
    assert "@media (max-width: 1280px)" in css
    assert "margin-right: min(490px, 42vw)" in css
    assert "margin-right: min(160px, 28vw)" in css
    # Right-cluster chrome collapses so tabs keep a usable scrollport.
    assert ".sb-right .conn-status { display: none; }" in css
    assert ".sb-right .status-pill { display: none; }" in css


def test_rag_docs_table_scroll_and_column_classes() -> None:
    """RAG inventory must use fixed columns + scroll shell (not 110px ref-table first col)."""
    css = _CSS.read_text(encoding="utf-8")
    assert ".table-scroll" in css
    assert "#rag-docs-table" in css
    assert ".rag-col-name" in css
    assert "min-width: 36rem" in css

    rag = _RAG.read_text(encoding="utf-8")
    assert "table-scroll rag-docs-scroll" in rag
    assert 'id="rag-docs-table"' in rag
    assert "rag-col-name" in rag
    assert "cell-wrap" in rag
    assert "rag-hits-table" in rag or "rag-hit-source" in rag


def test_css_cache_bust_bumped() -> None:
    base = _BASE.read_text(encoding="utf-8")
    assert "app.css?v=17" in base
