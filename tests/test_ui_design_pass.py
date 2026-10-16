"""Regression pins for the design/usability pass (static template/CSS checks)."""
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
    assert base.index("tokens.css?v=1") < base.index("base.css?v=1") < base.index("components.css?v=1") < base.index("layout.css?v=1") < base.index("app.css?v=104") < base.index("typography.css?v=1")
    assert ".hf-card .meta .stat" in css
    assert "min-width: 40rem" in css or "min-width:40rem" in css


def test_base_cache_bust():
    assert "app.css?v=104" in _r("templates/base.html")


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


def test_project_overview_no_duplicate_topbar():
    assert "block topbar_right" not in _r("templates/project.html")


def test_settings_tour_step_count():
    assert "7-step" in _r("templates/settings.html")
