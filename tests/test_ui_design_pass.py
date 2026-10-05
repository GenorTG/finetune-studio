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
    assert "var(--bg-nav)" in css
    assert ':root[data-theme="light"]' in css
    assert ".hf-card .meta .stat" in css
    assert "min-width: 40rem" in css or "min-width:40rem" in css


def test_base_cache_bust():
    assert "app.css?v=76" in _r("templates/base.html")


def test_project_overview_no_duplicate_topbar():
    assert "block topbar_right" not in _r("templates/project.html")


def test_settings_tour_step_count():
    assert "7-step" in _r("templates/settings.html")
