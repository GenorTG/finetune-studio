"""Regression pins for the design/usability pass (static template/CSS checks)."""
import re
from pathlib import Path

W = Path(__file__).resolve().parents[1] / "src" / "finetune_studio" / "webui"


def _r(rel):
    return (W / rel).read_text(encoding="utf-8")


def test_no_first_paint_boot_toast():
    js = _r("static/js/sprites.js")
    assert "spriteBoot()" not in js.replace("function spriteBoot()", "")


def test_css_pins():
    css = _r("static/css/app.css")
    tokens = _r("static/css/tokens.css")
    base = _r("templates/base.html")
    assert "var(--bg-nav)" in css
    assert ':root[data-theme="light"]' in tokens
    assert ':root[data-theme="light"]' not in base
    assert base.index("tokens.css?v=1") < base.index("base.css?v=1") < base.index("components.css?v=2") < base.index("layout.css?v=1") < base.index("app.css?v=118") < base.index("typography.css?v=1")
    assert ".hf-card .meta .stat" in css
    assert "min-width: 40rem" in css or "min-width:40rem" in css


def test_base_cache_bust():
    assert "app.css?v=118" in _r("templates/base.html")


def test_project_data_modals_use_hidden_state():
    template = _r("templates/project_data.html")
    assert 'id="fb-usage-modal" hidden' in template
    assert 'id="wb-editor-modal" hidden' in template
    assert 'id="wb-pane-parsed" hidden' in template
    assert "modal.hidden = false" in template
    assert "wbModal().hidden = true" in template
    assert "usageModal.hidden = true" in template
    assert "o.style.display" not in template and "p.style.display" not in template


def test_conditional_panels_use_hidden_state():
    exports = _r("templates/export_models.html")
    testing = _r("templates/project_testing.html")
    brain = _r("templates/brain.html")
    settings = _r("templates/settings.html")
    settings_js = _r("static/js/settings.js")

    assert 'id="export-result" class="p-2 pt-0" hidden' in exports
    assert "$('export-result').hidden = true" in exports
    assert 'id="t-suite-hint" class="text-xs dim mt-1" hidden' in testing
    assert "hint.hidden = false" in testing and "hint.hidden = true" in testing
    assert 'id="mcp-pass-box" hidden' in brain
    assert "box.hidden = false" in brain and "box.hidden = true" in brain
    assert 'id="update-log" class="text-xs" hidden' in settings
    assert "el.hidden = true" in settings_js and "el.hidden = false" in settings_js


def test_grid_full_utility_replaces_inline_spans():
    css = _r("static/css/app.css")
    assert ".grid-full { grid-column: 1 / -1; }" in css
    for name in ("projects", "project_training", "project_models", "export_models", "hf_models"):
        assert "grid-column:" not in _r(f"templates/{name}.html"), name


def test_settings_reuses_shared_max_width_utility():
    settings = _r("templates/settings.html")
    css = _r("static/css/app.css")
    assert ".max-w-420 { max-width: 420px; }" in css
    assert settings.count("max-width:420px") == 0


def test_templates_use_spacing_utilities_for_static_margins():
    for name in ("export_models", "projects", "project_models", "data_prep"):
        template = _r(f"templates/{name}.html")
        assert 'style="margin-top:12px' not in template
        assert 'style="margin-top:8px' not in template
    assert "wrap mt-3" in _r("templates/export_models.html")
    assert "dp-items-scroll mt-2" in _r("templates/data_prep.html")


def test_templates_use_shared_flex_utilities():
    css = _r("static/css/app.css")
    assert ".flex-1 { flex: 1; }" in css
    assert ".min-w-0 { min-width: 0; }" in css
    for name in ("projects", "hf_models", "project", "index", "inference", "brain"):
        template = _r(f"templates/{name}.html")
        assert 'style="flex:1;' not in template


def test_templates_reuse_full_width_utility():
    css = _r("static/css/app.css")
    assert ".w-full { width: 100%; }" in css
    for name in ("chat_v2", "benchmarks", "project_testing", "project_wizard", "brain", "project_compare"):
        template = _r(f"templates/{name}.html")
        assert 'style="width:100%;' not in template


def test_templates_reuse_auto_margin_utility():
    css = _r("static/css/app.css")
    assert ".ml-auto { margin-left: auto; }" in css
    for name in ("projects", "hf_models", "project", "project_data", "data_editor", "project_testing", "brain", "project_compare"):
        assert 'style="margin-left:auto' not in _r(f"templates/{name}.html"), name


def test_data_prep_reuses_bottom_margin_utility():
    css = _r("static/css/app.css")
    template = _r("templates/data_prep.html")
    assert ".mb-2 { margin-bottom: 8px; }" in css
    assert 'style="margin-bottom:8px' not in template
    assert 'class="text-xs dim mb-2" id="dp-grounded-note"' in template


def test_templates_reuse_empty_state_spacing():
    css = _r("static/css/app.css")
    assert ".center { text-align: center; }" in css
    assert ".p-5   { padding: 1.25rem; }" in css
    assert ".p-6   { padding: 1.5rem; }" in css
    assert ".p-8   { padding: 2rem; }" in css
    for name in ("project", "project_data", "data_prep", "data_editor", "models_index"):
        template = _r(f"templates/{name}.html")
        assert 'style="text-align:center;padding:' not in template
        assert 'style="padding:24px;text-align:center' not in template


def test_templates_reuse_margin_and_pointer_utilities():
    css = _r("static/css/app.css")
    assert r".mt-1\.5 { margin-top: 0.375rem; }" in css
    assert ".m-0   { margin: 0; }" in css
    assert ".cursor-pointer { cursor: pointer; }" in css
    assert 'style="margin:0"' not in _r("templates/benchmarks.html")
    assert 'style="margin:0"' not in _r("templates/project_wizard.html")
    for name in ("data_prep", "index"):
        assert 'style="margin-top:6px' not in _r(f"templates/{name}.html")
    for name in ("project_data", "project_wizard", "brain"):
        assert 'style="cursor:pointer;"' not in _r(f"templates/{name}.html")


def test_file_previews_share_component_styles():
    components = _r("static/css/components.css")
    base = _r("templates/base.html")
    assert ".file-preview-image" in components
    assert ".file-preview-frame" in components
    assert ".file-preview-content" in components
    assert "components.css?v=2" in base
    for name in ("project", "project_data", "data_prep"):
        template = _r(f"templates/{name}.html")
        assert 'style="max-width:100%;max-height:' not in template
        assert 'style="width:100%;height:' not in template
        assert 'style="max-height:70vh;overflow:auto;padding:12px;' not in template
        assert 'style="max-height:60vh;overflow:auto;padding:12px;' not in template


def test_benchmark_actions_reuse_small_button_variant():
    template = _r("templates/benchmarks.html")
    assert template.count('class="btn sm') >= 3
    assert 'style="font-size:11px;padding:4px 10px;"' not in template


def test_chat_model_status_tracks_the_inference_worker_not_the_guide_helper():
    sprites = _r("static/js/sprites.js")
    block = sprites.split("// Test chat uses the Inference worker", 1)[1].split(
        "// CPU chip", 1
    )[0]
    assert "fetch('/api/inference/status')" in block
    assert "fetch('/api/providers')" not in block
    assert "INFERENCE MODEL LOADED" in block
    chat = _r("templates/chat_v2.html")
    assert "Test chat uses the model loaded into Inference—not the Guide/API helper." in chat


def test_project_overview_no_duplicate_topbar():
    assert "block topbar_right" not in _r("templates/project.html")


def test_settings_tour_step_count():
    assert "7-step" in _r("templates/settings.html")


def test_files_page_has_no_inline_styles():
    # Inline styles here silently beat app.css (the checkbox width made a 700px rule dead).
    assert 'style="' not in _r("templates/project_data.html")
    css = _r("static/css/app.css")
    for cls in (".modal-dialog.modal-sm", ".modal-dialog.modal-lg", ".modal-dialog.modal-xl", ".fb-cols-menu {"):
        assert cls in css, cls


def test_data_prep_inline_styles_only_for_data_driven_widths():
    lines = [ln for ln in _r("templates/data_prep.html").splitlines() if 'style="' in ln]
    assert len(lines) == 1 and "style=\"width:' +" in lines[0], lines


def test_template_inline_styles_are_only_data_driven():
    """Inline styles silently beat app.css; only runtime values may stay inline.

    The allowlist only shrinks: progress widths computed at runtime, the compare
    grid's column count, and the benchmark sample picker that JS shows/hides.
    """
    allowed = {
        ("benchmarks.html", 'style="display:none"'): 2,
        ("data_prep.html", "style=\"width:' + (100 * c[k] / n).toFixed(1) + '%\""): 1,
        ("inference.html", 'style="width:0%"'): 1,
        ("project_compare.html", "style=\"grid-template-columns:' + cols + '\""): 1,
        ("project_testing.html", "style=\"width:' + w + '%\""): 1,
        ("project_training.html", 'style="width:0%"'): 1,
        ("_resources.html", 'style="width:0%"'): 1,
        ("_resources.html", 'style="width:${Math.min(100, g.pct)}%"'): 1,
    }
    found: dict = {}
    for path in sorted((W / "templates").rglob("*.html")):
        for m in re.finditer(r'style="[^"]*"', path.read_text(encoding="utf-8")):
            key = (path.name, m.group(0))
            found[key] = found.get(key, 0) + 1
    assert found == allowed



def test_training_inline_style_only_for_js_progress_width():
    lines = [ln.strip() for ln in _r("templates/project_training.html").splitlines() if 'style="' in ln]
    assert lines == ['<div class="bar mt-3"><div class="fill" id="train-progress-bar" style="width:0%"></div></div>'], lines
    assert "sev-warning" in _r("templates/project_training.html") and ".sev-warning" in _r("static/css/app.css")


def test_data_editor_table_scrolls_on_phones_instead_of_crushing_qa():
    assert 'id="de-table"' in _r("templates/data_editor.html")
    css = _r("static/css/app.css")
    assert ".table-scroll #de-table { min-width: 1100px; }" in css
    assert "#de-table th.de-col-actions { width: 360px; }" in css


def test_overview_files_and_compute_tables_fit_on_small_screens():
    assert 'id="ov-files-table"' in _r("templates/project.html")
    css = _r("static/css/app.css")
    assert ".table-scroll #ov-files-table { min-width: 800px; }" in css
    assert "#compute-table td:nth-child(3) { white-space: normal; }" in css
