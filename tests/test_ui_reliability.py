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
    assert "font-size: 14px" in css
    assert ".ws-label" in css
    # Mid-width media query must not collapse labels to ≤11px again.
    assert ".sb-tab-label { font-size: 13px; }" in css
    assert ".sb-tab-label { font-size: 11px; }" not in css
    assert ".sb-tab-label { font-size: 10px; }" not in css
    # 780px floor keeps tabs readable (not ≤11px).
    assert "@media (max-width: 780px)" in css
    block_780 = css.split("@media (max-width: 780px)", 1)[1].split("@media", 1)[0]
    assert ".sb-tab-label { font-size: 13px; }" in block_780
    assert "font-size: 11px" not in block_780
    assert "font-size: 10px" not in block_780


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
    assert "#fb-files-table" in css
    assert ".fl-table { min-width: 36rem; }" in css

    dp = _DATA_PREP.read_text(encoding="utf-8")
    assert "table-layout: fixed" in dp
    assert 'class="fl-col-name"' in dp
    assert 'colspan="7"' in dp
    assert "empty-row" in dp
    assert "table-scroll" in dp

    pdata = _PROJECT_DATA.read_text(encoding="utf-8")
    assert 'colspan="8"' in pdata
    assert "empty-row" in pdata
    assert "cell-wrap" in pdata
    assert "table-scroll" in pdata
    assert "Uploaded files" in pdata


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
    # Gutter is padding-right (inside the box) — never margin-right + width:100%.
    assert "padding-right: min(490px, 42vw)" in css
    assert "padding-right: min(160px, 28vw)" in css
    assert "margin-right: min(490px, 42vw)" not in css
    assert "margin-right: min(160px, 28vw)" not in css
    # Right-cluster chrome collapses so tabs keep a usable scrollport.
    assert ".sb-right .conn-status { display: none; }" in css
    assert ".sb-right .status-pill { display: none; }" in css


def test_header_no_page_overflow_contract_at_640() -> None:
    """At ≤700px the session bar must not widen the document; tabs stay scrollable."""
    css = _CSS.read_text(encoding="utf-8")
    assert "overflow-x: clip" in css
    assert ".session-bar" in css
    assert "overflow-x: hidden" in css  # session-bar containment
    assert "min-width: 0" in css
    block_700 = css.split("@media (max-width: 700px)", 1)[1]
    # Prefer the stacked responsive block (second 700px query has sb-tabs rules).
    if ".sb-tabs {" in block_700.split("@media", 1)[0]:
        narrow = block_700.split("@media", 1)[0]
    else:
        # First 700px block is brand-only; take the later stacked one.
        parts = css.split("@media (max-width: 700px)")
        narrow = parts[-1].split("@media", 1)[0]
    assert "padding-right: 8px" in narrow
    assert "position: static" in narrow  # .sb-right stacked under tabs
    assert "font-size: 13px" in narrow
    assert "font-size: 11px" not in narrow
    # No width:100% + margin-right gutter (classic scrollWidth blowout).
    assert "margin-right: min(140px" not in narrow


def test_card_head_stacks_below_700() -> None:
    """Data Prep Uploaded-files card-head must stack title/actions below ~700px."""
    css = _CSS.read_text(encoding="utf-8")
    parts = css.split("@media (max-width: 700px)")
    assert len(parts) >= 2
    narrow = parts[-1].split("@media", 1)[0]
    assert ".card-head" in narrow
    assert "flex-direction: column" in narrow
    assert "align-items: stretch" in narrow
    # Base rule still flexes title/actions with a gap.
    assert ".card-head > :first-child" in css
    assert "min-width: 0" in css


def test_css_cache_bust_bumped() -> None:
    base = _BASE.read_text(encoding="utf-8")
    assert "app.css?v=21" in base
    assert "sprites.js?v=14" in base


def test_workflow_steps_vertical_at_desktop() -> None:
    """`.rag-workflow-steps` must be a vertical list at all widths (not flex-wrap row).

    At ~1280 the old wrap put steps 1–3 on one cramped line with markers colliding.
    Base rule must force column; mobile ≤780 may still tighten gap/font.
    """
    css = _CSS.read_text(encoding="utf-8")
    # Base rule lives after media queries; pin the multi-line vertical contract.
    assert (
        ".rag-workflow-steps {\n"
        "  display: flex;\n"
        "  flex-direction: column;\n"
        "  flex-wrap: nowrap;"
    ) in css
    # Must not regress to the horizontal wrap that cramped desktop ~1280.
    assert "flex-wrap: wrap;\n  gap: 0.35rem 1.1rem;" not in css
    # Mobile override keeps column (readable sequence preserved).
    block_780 = css.split("@media (max-width: 780px)", 1)[1].split("@media", 1)[0]
    assert ".rag-workflow-steps" in block_780
    assert "flex-direction: column" in block_780

    dp = _DATA_PREP.read_text(encoding="utf-8")
    assert 'id="dp-workflow"' in dp
    assert 'class="rag-workflow-steps"' in dp
    assert 'aria-label="Data prep workflow"' in dp


def test_rag_docs_table_scroll_and_column_classes() -> None:
    """RAG inventory must use fixed columns + scroll shell (not 110px ref-table first col)."""
    css = _CSS.read_text(encoding="utf-8")
    assert ".table-scroll" in css
    assert "#rag-docs-table" in css
    assert ".rag-col-name" in css
    assert "min-width: 48rem" in css
    assert ".rag-col-actions" in css
    assert "min-width: 15rem" in css
    assert ".rag-doc-actions" in css
    assert "inline-flex" in css
    # Actions must not clip under overflow:hidden from generic .table td.
    assert "td.rag-col-actions" in css
    assert "overflow: visible" in css

    rag = _RAG.read_text(encoding="utf-8")
    assert "table-scroll rag-docs-scroll" in rag
    assert 'id="rag-docs-table"' in rag
    assert "rag-col-name" in rag
    assert "cell-wrap" in rag
    assert "rag-doc-actions" in rag
    assert "flex gap-1 rag-col-actions" not in rag
    assert "rag-hits-table" in rag or "rag-hit-source" in rag


def test_rag_ia_workflow_and_cta(client: TestClient) -> None:
    """RAG page must explain Upload→…→Test and link to Data Prep (no fake upload)."""
    pid = _project(client)
    r = client.get(f"/projects/{pid}/rag")
    assert r.status_code == 200, r.text
    body = r.text
    assert 'id="rag-workflow"' in body
    assert "Upload" in body and "Parse" in body and "Embed" in body and "Test" in body
    assert 'id="rag-cta-data-prep"' in body
    assert f'href="/projects/{pid}/data-prep"' in body
    assert "Data Prep / Upload Sources" in body
    assert "Indexed corpus documents" in body
    assert "Build with AI" not in body
    assert "agentic" not in body.lower()
    assert "Upload new files on Data Prep" in body
    assert "model weights" in body.lower()
    assert "No documents indexed yet" in body
    assert f"/projects/{pid}/data-prep" in body
    # Numbered sections are unique and sequential
    for title in (
        "1. Corpus status",
        "2. Build &amp; rebuild (embed)",
        "3. Retrieval settings",
        "4. Indexed corpus documents",
        "5. Live test query",
        "6. Chat with RAG",
        "7. Self-contained export",
        "8. Shared embedder / reranker library",
    ):
        assert title in body, title
    assert body.count("5. Live test query") == 1
    assert "5. Chat with RAG" not in body


def test_data_prep_ia_sections(client: TestClient) -> None:
    pid = _project(client)
    r = client.get(f"/projects/{pid}/data-prep")
    assert r.status_code == 200, r.text
    body = r.text
    assert 'id="dp-uploaded-files"' in body
    assert "Uploaded files" in body
    assert 'id="dp-parsed-sources"' in body
    assert "Parsed sources" in body
    assert "Training / Q&amp;A output" in body or "Training / Q&A output" in body
    assert 'id="dp-parsed-empty"' in body or "No parsed sources yet" in body
    assert f'href="/projects/{pid}/rag"' in body
    assert "flRenderParsedList" in body or "flRefreshSources" in body


def test_rag_workspace_active_on_rag_page(client: TestClient) -> None:
    pid = _project(client)
    r = client.get(f"/projects/{pid}/rag")
    assert r.status_code == 200, r.text
    body = r.text
    assert 'class="ws-switch active"' in body or "ws-switch active" in body
    assert 'aria-current="page"' in body
    assert "RAG workspace" in body
    assert "Model workspace" in body


def test_sprites_rag_caption_not_embedding_corpus() -> None:
    sprites = (
        _ROOT / "src" / "finetune_studio" / "webui" / "static" / "js" / "sprites.js"
    ).read_text(encoding="utf-8")
    assert "EMBEDDING CORPUS" not in sprites
    assert "RAG RETRIEVAL INDEX" in sprites


def test_sprites_bench_idle_caption_not_computing() -> None:
    """Bench/testing idle mount must not claim scores are computing."""
    sprites = (
        _ROOT / "src" / "finetune_studio" / "webui" / "static" / "js" / "sprites.js"
    ).read_text(encoding="utf-8")
    assert "COMPUTING SCORES" not in sprites
    assert "READY TO RUN" in sprites
    assert "fts:bench-progress" in sprites
    assert "ev.detail.score" in sprites
    assert "benchUpdate" in sprites
    assert "SCORE" in sprites
