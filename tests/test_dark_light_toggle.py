"""Regression tests for dark/light theme toggle (QABUG-008)."""

from __future__ import annotations

from pathlib import Path

_TEMPLATES = (
    Path(__file__).resolve().parents[1]
    / "src"
    / "finetune_studio"
    / "webui"
    / "templates"
)
_BASE = _TEMPLATES / "base.html"
_APP_JS = (
    Path(__file__).resolve().parents[1]
    / "src"
    / "finetune_studio"
    / "webui"
    / "static"
    / "js"
    / "app.js"
)


def test_base_html_has_theme_toggle_button() -> None:
    html = _BASE.read_text(encoding="utf-8")
    assert 'id="theme-toggle"' in html
    assert "ftsThemeToggle" in html or "theme-toggle" in html


def test_base_html_default_theme_is_dark() -> None:
    html = _BASE.read_text(encoding="utf-8")
    # <html> must not hardcode light; pre-paint script defaults to dark.
    head = html.split("<body", 1)[0]
    assert "<html" in head
    assert 'html lang="en" data-theme="light"' not in head
    assert "fts-theme" in html
    assert "t = 'dark'" in html


def test_base_html_persists_theme_in_localstorage() -> None:
    html = _BASE.read_text(encoding="utf-8")
    js = _APP_JS.read_text(encoding="utf-8")
    assert "localStorage.getItem('fts-theme')" in html or 'localStorage.getItem("fts-theme")' in html
    assert "fts-theme" in js
    assert "localStorage.setItem" in js
    assert "setAttribute('data-theme'" in js or 'setAttribute("data-theme"' in js
