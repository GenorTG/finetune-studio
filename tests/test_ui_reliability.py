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
_RAG = _ROOT / "src" / "finetune_studio" / "webui" / "templates" / "brain.html"
_TESTING = _ROOT / "src" / "finetune_studio" / "webui" / "templates" / "project_testing.html"
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
    assert 'aria-label="Project flow"' in base
    assert "1 Files &amp; Knowledge" in base and "2 Smart Brain" in base and "2 train" in base


def test_project_file_page_points_to_the_next_workflow_step(client: TestClient) -> None:
    pid = _project(client)
    response = client.get(f"/projects/{pid}/data")
    assert response.status_code == 200
    assert 'aria-label="Next step"' in response.text
    flow = response.text.split('id="workspace-subnav"', 1)[1].split("</nav>", 1)[0]
    assert 'aria-label="Next step"' in flow
    assert f'href="/projects/{pid}/data-prep"' in response.text
    assert "Review question-answer pairs" in response.text


def test_project_home_shows_live_checklist_and_single_primary_action(client: TestClient) -> None:
    pid = _project(client)
    response = client.get(f"/projects/{pid}")
    assert response.status_code == 200
    body = response.text
    checklist = body.split('id="project-checklist"', 1)[1].split("</section>", 1)[0]
    assert 'id="project-next-action"' in checklist
    assert f'href="/projects/{pid}/data"' in checklist
    assert "Add source files →" in checklist
    assert checklist.count('class="btn primary"') == 1
    assert "flow-card" not in body
    assert f'href="/projects/{pid}/brain"' in checklist


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
    assert 'colspan="9"' in pdata
    assert "empty-row" in pdata
    assert "cell-wrap" in pdata
    assert "table-scroll" in pdata
    assert "Uploaded files" in pdata


def test_data_prep_uploaded_files_mobile_column_floors() -> None:
    """At ≤780px, uploaded-files columns need rem floors + scroll affordance.

    Regression for 375px: percentage widths + max-width:0 collapsed columns
    to ~4–12px while the page itself did not overflow.
    """
    css = _CSS.read_text(encoding="utf-8")
    assert "#dp-uploaded-files .fl-col-name { width: 10rem; min-width: 10rem; }" in css
    assert "#dp-uploaded-files .fl-col-actions { width: 10rem; min-width: 10rem; }" in css
    assert "#dp-uploaded-files .fl-table" in css
    assert "min-width: 36rem" in css
    assert "#dp-uploaded-files .fl-name" in css
    assert "max-width: none" in css
    # Horizontal scroll lives on .table-scroll inside the drop zone.
    assert "#dp-uploaded-files .table-scroll" in css
    assert "overflow-x: auto" in css
    dp = _DATA_PREP.read_text(encoding="utf-8")
    assert 'id="dp-uploaded-files"' in dp
    assert 'class="table-scroll"' in dp
    assert 'class="fl-table"' in dp
    assert 'class="fl-col-name"' in dp
    assert 'class="fl-col-actions"' in dp


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
    assert "/brain/settings" in src


def test_inference_ui_requires_loaded_confirmation() -> None:
    src = _INFERENCE.read_text(encoding="utf-8")
    # api.post throws on the 4xx/5xx a failed load returns; success is still confirmed via status.
    assert "await api.post('/api/models/load'" in src
    assert "/api/inference/status" in src
    assert "still unloaded" in src or "still shows unloaded" in src


def test_load_missing_path_returns_explicit_failure(client: TestClient) -> None:
    r = client.post("/api/models/load", json={})
    assert r.status_code == 400, r.text
    data = r.json()
    assert data.get("status") == "error"
    assert data.get("loaded") is False
    assert data.get("detail")
    assert data.get("model") is None


def test_load_exception_returns_failure_not_loaded(
    client: TestClient, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    model_file = tmp_path / "missing-model.bin"
    model_file.write_bytes(b"x")
    engine = MagicMock()
    engine.model = None
    engine.vision = False

    def _boom(*_a, **_k):
        raise RuntimeError("CUDA OOM: tried to allocate 12 GiB")

    engine.load.side_effect = _boom
    monkeypatch.setattr("finetune_studio.webui.app.inference_engine", engine)

    r = client.post("/api/models/load", json={"path": str(model_file)})
    assert r.status_code == 500, r.text
    data = r.json()
    assert data["status"] == "error"
    assert data["loaded"] is False
    assert "CUDA OOM" in data["detail"]
    assert data.get("model") is None


def test_load_success_requires_model_object(
    client: TestClient, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """If load() returns without holding a model, never claim status=loaded."""
    engine = MagicMock()
    engine.model = None
    engine.vision = False
    engine.load = MagicMock()
    monkeypatch.setattr("finetune_studio.webui.app.inference_engine", engine)

    r = client.post("/api/models/load", json={"path": str(tmp_path)})
    assert r.status_code == 500, r.text
    data = r.json()
    assert data["status"] == "error"
    assert data["loaded"] is False
    assert "no model is held" in data["detail"]


def test_load_success_payload_when_model_held(
    client: TestClient, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    engine = MagicMock()
    engine.model = object()
    engine.vision = False
    engine.load = MagicMock()
    monkeypatch.setattr("finetune_studio.webui.app.inference_engine", engine)

    r = client.post("/api/models/load", json={"path": str(tmp_path)})
    assert r.status_code == 200, r.text
    data = r.json()
    assert data["status"] == "loaded"
    assert data["loaded"] is True
    assert data["model"] == str(tmp_path)
    assert engine.load.call_args.kwargs["n_gpu_layers"] == -1


def test_load_honours_an_explicit_gpu_layer_count_as_an_upper_bound(
    client: TestClient, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """The Inference page's "GPU offload layers" slider used to be ignored (always all layers). An explicit
    value is now passed through; -1 / absent / the legacy 99 mean "as many as fit"."""
    model_file = tmp_path / "ok.gguf"
    model_file.write_bytes(b"x")
    engine = MagicMock()
    engine.model = object()
    engine.vision = False
    engine.offload = {"mode": "partial", "gpu_layers": 24, "total_layers": 36, "warnings": ["24/36 layers"]}
    monkeypatch.setattr("finetune_studio.webui.app.inference_engine", engine)

    for requested, expected in ((0, 0), (24, 24), (-1, -1), (99, -1)):
        response = client.post(
            "/api/models/load",
            json={"path": str(model_file), "n_gpu_layers": requested},
        )
        assert response.status_code == 200, response.text
        assert engine.load.call_args.kwargs["n_gpu_layers"] == expected
    body = response.json()
    assert body["offload"]["mode"] == "partial" and body["warnings"] == ["24/36 layers"]


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
    base_css = (_ROOT / "src" / "finetune_studio" / "webui" / "static" / "css" / "base.css").read_text(encoding="utf-8")
    assert "overflow-x: clip" in base_css
    assert ".session-bar" in css
    assert "overflow-x: hidden" in css  # session-bar containment
    assert "min-width: 0" in css
    block_700 = css.split("@media (max-width: 700px)", 1)[1]
    # Prefer the stacked responsive block (shell + tabs rules).
    head = block_700.split("@media", 1)[0]
    if ".sb-tabs {" in head or ".sb-nav-shell {" in head:
        narrow = head
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
    assert ".sb-nav-shell" in narrow or ".sb-tabs" in narrow


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
    assert "app.css?v=108" in base
    assert "sprites.js?v=20" in base
    assert "nav_overflow.js?v=2" in base


def test_testing_page_styles_live_in_versioned_stylesheet() -> None:
    template = _TESTING.read_text(encoding="utf-8")
    css = _CSS.read_text(encoding="utf-8")
    base = _BASE.read_text(encoding="utf-8")
    assert "<style>" not in template
    assert ".tp-seg" in css and ".tp-runs" in css and ".rv-main" in css
    assert 'app.css?v=108' in base
    assert "spa.js?v=19" in base


def test_workflow_steps_vertical_at_desktop() -> None:
    """`.brain-workflow-steps` must be a vertical list at all widths (not flex-wrap row).

    At ~1280 the old wrap put steps 1–3 on one cramped line with markers colliding.
    Base rule must force column; mobile ≤780 may still tighten gap/font.
    """
    css = _CSS.read_text(encoding="utf-8")
    # Base rule lives after media queries; pin the multi-line vertical contract.
    assert (
        ".brain-workflow-steps {\n"
        "  display: flex;\n"
        "  flex-direction: column;\n"
        "  flex-wrap: nowrap;"
    ) in css
    # Must not regress to the horizontal wrap that cramped desktop ~1280.
    assert "flex-wrap: wrap;\n  gap: 0.35rem 1.1rem;" not in css
    # Mobile override keeps column (readable sequence preserved).
    block_780 = css.split("@media (max-width: 780px)", 1)[1].split("@media", 1)[0]
    assert ".brain-workflow-steps" in block_780
    assert "flex-direction: column" in block_780

    dp = _DATA_PREP.read_text(encoding="utf-8")
    assert 'id="dp-workflow"' in dp
    assert 'class="brain-workflow-steps"' in dp
    assert 'aria-label="Data prep workflow"' in dp


def test_brain_docs_table_scroll_and_column_classes() -> None:
    """RAG inventory must use fixed columns + scroll shell (not 110px ref-table first col)."""
    css = _CSS.read_text(encoding="utf-8")
    assert ".table-scroll" in css
    assert "#brain-docs-table" in css
    assert ".brain-col-name" in css
    assert "min-width: 48rem" in css
    assert ".brain-col-actions" in css
    assert "min-width: 15rem" in css
    assert ".brain-doc-actions" in css
    assert "inline-flex" in css
    # Actions must not clip under overflow:hidden from generic .table td.
    assert "td.brain-col-actions" in css
    assert "overflow: visible" in css

    rag = _RAG.read_text(encoding="utf-8")
    assert "table-scroll brain-docs-scroll" in rag
    assert 'id="brain-docs-table"' in rag
    assert "brain-col-name" in rag
    assert "cell-wrap" in rag
    assert "brain-doc-actions" in rag
    assert "flex gap-1 brain-col-actions" not in rag
    assert "brain-hits-table" in rag or "brain-hit-source" in rag


def test_brain_ia_workflow_and_cta(client: TestClient) -> None:
    """Smart Brain page explains where it sits (Files & Knowledge → build → test) and has no upload of its own."""
    pid = _project(client)
    r = client.get(f"/projects/{pid}/brain")
    assert r.status_code == 200, r.text
    body = r.text
    assert '<details class="card mb-4" id="brain-workflow">' in body
    assert '<summary class="card-head">' in body
    assert "Upload" in body and "parsed on arrival" in body and "Build" in body and "Test" in body
    assert 'id="brain-cta-data"' in body and f'href="/projects/{pid}/data"' in body
    assert "Files &amp; Knowledge" in body
    assert "Methods:" not in body
    more = body.split('<details class="page-more">', 1)[1].split("</details>", 1)[0]
    assert "Search combines multilingual semantic embeddings" in more
    assert "<h1>🧠 Smart Brain</h1>" in body and "RAG index" not in body
    assert "Build with AI" not in body
    assert "agentic" not in body.lower()
    assert "Upload new files on the files page" in body
    assert body.count('id="quick-index-btn"') == 1
    assert body.index('id="quick-index-btn"') < body.index('id="brain-workflow"')
    assert "Use Quick index above" in body
    assert "model weights" in body.lower()
    assert "No documents indexed yet" in body
    # Numbered sections are unique and sequential
    for title in (
        "1. Index status",
        "2. Build the index",
        "3. Search settings",
        "4. Documents in the index",
        "5. Search test",
        "6. Ask the model (grounded)",
        "7. Download / export the index",
        "8. Shared search models",
    ):
        assert title in body, title
    assert body.count("5. Search test") == 1
    assert "Chat with RAG" not in body
    assert '<details class="card mb-4" id="brain-search-settings">' in body
    assert '<details class="card mb-4" id="brain-docs-panel">' in body
    assert '<details class="card mb-4" id="brain-export-panel">' in body
    assert '<details class="card mb-4" id="brain-shared-models">' in body


def test_data_prep_ia_sections(client: TestClient) -> None:
    pid = _project(client)
    r = client.get(f"/projects/{pid}/data-prep")
    assert r.status_code == 200, r.text
    body = r.text
    assert 'id="dp-uploaded-files"' in body
    assert '<details class="card mb-6" id="dp-uploaded-files">' in body
    assert '<summary class="card-head">' in body
    assert 'id="dp-uploaded-count"' in body
    assert "count.textContent = _flTotalCount" in body
    assert ".dp-points-col { min-width: 12rem; white-space: nowrap; }" in body
    assert '<details class="card mb-6" id="dp-pref-card">' in body
    assert 'details.card:not([open]) > summary.card-head' in _CSS.read_text(encoding="utf-8")
    assert "Uploaded files" in body
    assert 'id="dp-mining"' in body
    assert "Mine pairs, file by file" in body
    assert "Training / Q&amp;A output" in body or "Training / Q&A output" in body
    assert 'id="dp-files-empty"' in body  # the empty-state text the files table renders
    assert f'href="/projects/{pid}/brain"' in body
    assert "dpRenderFiles" in body


def test_settings_and_inference_pages_have_headings(client: TestClient) -> None:
    for path, heading in (("/settings", "Settings"), ("/inference", "Inference")):
        response = client.get(path)
        assert response.status_code == 200, response.text
        assert f"<h1>{heading}</h1>" in response.text


def test_brain_is_marked_here_in_the_flow_bar(client: TestClient) -> None:
    pid = _project(client)
    r = client.get(f"/projects/{pid}/brain")
    assert r.status_code == 200, r.text
    bar = r.text.split('id="workspace-subnav"', 1)[1].split("</nav>", 1)[0]
    assert "ws-switch" not in bar                                   # no model/RAG workspace switch any more
    here = bar.split('data-tab="brain"', 1)[1].split(">", 1)[0]
    assert "here" in here and "2 Smart Brain" in bar and "2 train" in bar


def test_sprites_brain_caption_not_embedding_corpus() -> None:
    sprites = (
        _ROOT / "src" / "finetune_studio" / "webui" / "static" / "js" / "sprites.js"
    ).read_text(encoding="utf-8")
    assert "EMBEDDING CORPUS" not in sprites
    assert "SMART BRAIN INDEX" in sprites


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


def test_sprites_training_idle_not_always_accelerating() -> None:
    """Training page must show READY TO TRAIN while idle, not GPU ACCELERATING."""
    sprites = (
        _ROOT / "src" / "finetune_studio" / "webui" / "static" / "js" / "sprites.js"
    ).read_text(encoding="utf-8")
    train_block = sprites.split("CPU chip → training", 1)[1].split(
        "Data stream → data prep", 1
    )[0]
    assert "READY TO TRAIN" in train_block
    assert "/api/training/status" in train_block
    assert "GPU ACCELERATING" in train_block  # only when status is training/running
    assert "LOADING MODEL" in train_block
    # Idle READY only when the form can actually start (audit 2026-10-03).
    assert "NEEDS A BASE MODEL" in train_block and "PICK A DATASET" in train_block
    assert "train-base-model" in train_block and "data-path-input" in train_block
    assert "SAVING" in train_block
    # Must not hard-code accelerating as the initial caption.
    assert "cap.textContent = 'GPU ACCELERATING';" not in train_block.split(
        "refreshTrainCaption", 1
    )[0]
    assert "_pageSpriteCleanup" in sprites
    assert "document.querySelectorAll('.sprite-mount')" in sprites


def test_file_library_all_files_uses_total_count() -> None:
    """Tree 'all files' badge must use API total_count, not filtered _flFiles.length."""
    src = _DATA_PREP.read_text(encoding="utf-8")
    assert "_flTotalCount" in src
    assert "total_count" in src
    assert "id=\"fl-all-count\"" in src or "id='fl-all-count'" in src
    all_row = src.split("all files</a>", 1)[1].split("</li>", 1)[0]
    assert "_flTotalCount" in all_row
    assert "_flFiles.length" not in all_row
    load_js = src.split("async function flLoadFiles", 1)[1].split(
        "async function flLoadTrash", 1
    )[0]
    assert "flRenderFolders()" in load_js
    assert "r.total_count" in load_js


def test_inference_model_info_uses_get() -> None:
    """Client must GET /api/models/info?path=… — POST is not allowed."""
    src = _INFERENCE.read_text(encoding="utf-8")
    assert "fts.api.post('/api/models/info'" not in src
    assert "fts.api.get('/api/models/info?path='" in src
    assert "encodeURIComponent(path)" in src
    assert "Model info failed" in src
    assert "total_layers || meta.num_layers" in src


def test_models_info_get_contract(client: TestClient, tmp_path: Path) -> None:
    """GET /api/models/info?path= works; POST must not silently succeed as GET."""
    model_dir = tmp_path / "tiny-model"
    model_dir.mkdir()
    (model_dir / "config.json").write_text(
        '{"architectures":["Qwen3ForCausalLM"],"model_type":"qwen3",'
        '"hidden_size":64,"num_hidden_layers":12,"vocab_size":100}',
        encoding="utf-8",
    )
    r = client.get("/api/models/info", params={"path": str(model_dir)})
    assert r.status_code == 200, r.text
    data = r.json()
    assert data.get("path") == str(model_dir)
    assert data.get("num_layers") == 12
    assert data.get("total_layers") == 12

    post = client.post("/api/models/info", json={"path": str(model_dir)})
    assert post.status_code == 405, post.text


def test_training_start_disabled_until_dataset(client: TestClient) -> None:
    """Start training must look disabled until a dataset/data_path is chosen."""
    pid = _project(client)
    r = client.get(f"/projects/{pid}/training")
    assert r.status_code == 200, r.text
    body = r.text
    assert 'id="start-btn"' in body
    assert "disabled" in body.split('id="start-btn"', 1)[1].split(">", 1)[0]
    assert "syncStartEnabled" in body
    assert "Pick or upload a dataset first" in body
    assert "~1.5 GB" not in body
    assert "roughly the size of the base model" in body
    # Trainable-base filtering kept (server models_for_training + HF merge skips GGUF).
    assert "trainable HF weights" in body
    assert "hasGguf" in body
    # Stop stays disabled while idle (no active run).
    assert 'id="stop-btn"' in body
    assert "disabled" in body.split('id="stop-btn"', 1)[1].split(">", 1)[0]
    assert "No active training run" in body

_MODELS_INDEX = (
    _ROOT / "src" / "finetune_studio" / "webui" / "templates" / "models_index.html"
)
_BENCH = _ROOT / "src" / "finetune_studio" / "webui" / "templates" / "benchmarks.html"
_HF = _ROOT / "src" / "finetune_studio" / "webui" / "templates" / "hf_models.html"


def test_bench_mobile_table_scroll_contract() -> None:
    """375px must keep RUN actions reachable via explicit horizontal scroll."""
    css = _CSS.read_text(encoding="utf-8")
    bench = _BENCH.read_text(encoding="utf-8")
    assert 'id="bench-run-scroll"' in bench
    assert "bench-table-scroll" in bench
    assert "min-width: 6.5rem" in css
    assert "#bench-run-table" in css
    assert "@media (max-width: 780px)" in css
    assert ".models-table-scroll" in css
    assert "overflow-x: auto" in css


def test_models_index_mobile_table_scroll_contract() -> None:
    """Live /models renders models_index.html — must scroll, not crush Copy to 2ch."""
    css = _CSS.read_text(encoding="utf-8")
    body = _MODELS_INDEX.read_text(encoding="utf-8")
    assert 'id="models-index-table-scroll"' in body
    assert 'class="table-scroll models-table-scroll"' in body
    assert 'id="models-index-table"' in body
    assert 'class="models-col-actions"' in body
    assert "filterModels" in body
    assert "#models-index-table tr[data-category]" in body
    assert "#models-index-filters" in body
    assert "#models-index-table" in css
    assert "#models-index-table .models-col-actions" in css
    assert "min-width: 52rem" in css
    assert ".models-table-scroll" in css
    assert "overflow-x: auto" in css


def test_data_toolbar_actions_wrap_on_mobile() -> None:
    css = _CSS.read_text(encoding="utf-8")
    pdata = _PROJECT_DATA.read_text(encoding="utf-8")
    assert 'id="fb-toolbar-actions"' in pdata
    assert "Move to folder" in pdata
    assert "#fb-toolbar-actions" in css
    assert "flex-wrap: wrap" in css
    assert "min-width: 7.5rem" in css


def test_inference_load_button_reachable_on_mobile() -> None:
    """Fixed-height desktop grid must not clip Load model at ≤780px."""
    css = _CSS.read_text(encoding="utf-8")
    inf = _INFERENCE.read_text(encoding="utf-8")
    assert 'id="inference-layout"' in inf
    assert "inference-layout" in inf
    assert 'id="inference-load-actions"' in inf
    assert 'id="load-btn"' in inf
    assert "height: calc(100vh" not in inf  # moved to CSS class
    assert ".inference-layout" in css
    # Mobile override clears fixed height so the load actions stay in flow.
    assert "height: auto !important" in css
    assert ".inference-load-actions" in css


def test_hf_modal_escape_close_contract() -> None:
    html = _HF.read_text(encoding="utf-8")
    assert "function closeHfModal" in html
    assert "ev.key !== 'Escape'" in html or 'ev.key !== "Escape"' in html
    assert 'role="dialog"' in html
    assert 'aria-modal="true"' in html
    assert 'aria-label="Close"' in html
    assert "openHfModal" in html


_PROJECT_MODELS = (
    _ROOT / "src" / "finetune_studio" / "webui" / "templates" / "project_models.html"
)
_EXPORT_MODELS = (
    _ROOT / "src" / "finetune_studio" / "webui" / "templates" / "export_models.html"
)
_TRAINING = (
    _ROOT / "src" / "finetune_studio" / "webui" / "templates" / "project_training.html"
)


def test_project_and_export_trained_exports_mobile_scroll() -> None:
    """project_models + export_models trained-exports tables must scroll at 375px."""
    css = _CSS.read_text(encoding="utf-8")
    for path in (_PROJECT_MODELS, _EXPORT_MODELS):
        body = path.read_text(encoding="utf-8")
        assert 'id="trained-exports-table-scroll"' in body, path.name
        assert "trained-exports-table-scroll" in body, path.name
        assert 'id="trained-exports-table"' in body, path.name
        assert "exports-col-actions" in body, path.name
        assert "exports-col-copy" in body, path.name
        assert "Copy path" in body, path.name
        assert "Open in inference" in body, path.name
    assert "#trained-exports-table" in css
    assert "#trained-exports-table .exports-col-actions" in css
    assert "min-width: 56rem" in css
    assert ".trained-exports-table-scroll" in css
    assert "overflow-x: auto" in css


def test_models_index_filters_discoverable_on_mobile() -> None:
    """Filter pills must wrap (or scroll) — .flex-wrap alone was undefined and clipped."""
    css = _CSS.read_text(encoding="utf-8")
    body = _MODELS_INDEX.read_text(encoding="utf-8")
    assert 'id="models-index-filters"' in body
    assert "models-index-filters" in body
    assert ".models-index-filters" in css or "#models-index-filters" in css
    assert "flex-wrap: wrap" in css
    # Explicit filter rule exists and allows horizontal scroll as backup.
    assert "#models-index-filters" in css
    filt_block = css.split("#models-index-filters", 1)[1][:400]
    assert "flex-wrap" in filt_block or "wrap" in css.split(".models-index-filters", 1)[-1][:500]
    assert "overflow-x: auto" in css.split(".models-index-filters", 1)[-1][:600] or (
        "overflow-x: auto" in css.split("#models-index-filters", 1)[-1][:600]
    )


def test_training_stop_disabled_while_idle() -> None:
    """Stop must be disabled on idle load and only enable while a run is active."""
    body = _TRAINING.read_text(encoding="utf-8")
    assert 'id="stop-btn"' in body
    stop_tag = body.split('id="stop-btn"', 1)[1].split(">", 1)[0]
    assert "disabled" in stop_tag
    assert "No active training run" in body
    assert "stopBtn.disabled" in body
    assert "ACTIVE[st]" in body


def test_training_routes_keep_sft_primary_and_fold_alternatives() -> None:
    body = _TRAINING.read_text(encoding="utf-8")
    assert '<legend class="text-xs muted">Training route</legend>' in body
    assert body.count('name="training_mode" value="sft"') == 1
    assert body.count('id="advanced-training-routes"') == 1
    assert body.index('value="sft"') < body.index('id="advanced-training-routes"')
    folded = body.split('id="advanced-training-routes"', 1)[1].split("</details>", 1)[0]
    for mode in ("dpo", "kto", "tool_sft", "continued_pretraining", "reasoning_distillation"):
        assert folded.count(f'name="training_mode" value="{mode}"') == 1
    assert 'querySelectorAll(\'input[name="training_mode"]\')' in body


def test_wizard_step5_not_ready_without_finished_run() -> None:
    """Quick work step 5 showed READY with zero runs; it now names what's missing,
    and treats the RAG coverage empty states (200 + state) as 'not built'."""
    wiz = (
        _ROOT / "src" / "finetune_studio" / "webui" / "templates" / "project_wizard.html"
    ).read_text(encoding="utf-8")
    assert '"needs a trained model"' in wiz and "res.nDone > 0" in wiz
    assert 'd.state === "no_sources" || d.state === "not_built"' in wiz
