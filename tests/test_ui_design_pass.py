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
    assert base.index("tokens.css?v=1") < base.index("base.css?v=1") < base.index("components.css?v=1") < base.index("layout.css?v=1") < base.index("app.css?v=100") < base.index("typography.css?v=1")
    assert ".hf-card .meta .stat" in css
    assert "min-width: 40rem" in css or "min-width:40rem" in css


def test_base_cache_bust():
    assert "app.css?v=100" in _r("templates/base.html")


def test_project_data_modals_use_hidden_state():
    template = _r("templates/project_data.html")
    assert 'id="fb-usage-modal" hidden' in template
    assert 'id="wb-editor-modal" hidden' in template
    assert 'id="wb-pane-parsed" hidden' in template
    assert "modal.hidden = false" in template
    assert "wbModal().hidden = true" in template
    assert "usageModal.hidden = true" in template
    assert "o.style.display" not in template and "p.style.display" not in template


def test_project_overview_no_duplicate_topbar():
    assert "block topbar_right" not in _r("templates/project.html")


def test_settings_tour_step_count():
    assert "7-step" in _r("templates/settings.html")
