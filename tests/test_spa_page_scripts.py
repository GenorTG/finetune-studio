"""Regression tests for SPA page-scripts injection (scripts block outside #content)."""

from __future__ import annotations

import re
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
_TEMPLATES = _ROOT / "src" / "finetune_studio" / "webui" / "templates"
_BASE = _TEMPLATES / "base.html"
_SPA_JS = _ROOT / "src" / "finetune_studio" / "webui" / "static" / "js" / "spa.js"


def test_base_html_wraps_scripts_block_in_page_scripts() -> None:
    html = _BASE.read_text(encoding="utf-8")
    assert 'id="page-scripts"' in html
    assert "{% block scripts %}{% endblock %}" in html
    # scripts block must live inside the page-scripts element
    m = re.search(
        r'<div[^>]*\bid=["\']page-scripts["\'][^>]*>.*?\{%\s*block\s+scripts\s*%\}',
        html,
        re.DOTALL,
    )
    assert m is not None, "page-scripts must wrap {% block scripts %}"


def test_page_scripts_comes_after_content() -> None:
    html = _BASE.read_text(encoding="utf-8")
    content_pos = html.find('id="content"')
    scripts_pos = html.find('id="page-scripts"')
    assert content_pos != -1
    assert scripts_pos != -1
    assert scripts_pos > content_pos


def test_spa_js_references_page_scripts_and_rewrites_decls() -> None:
    js = _SPA_JS.read_text(encoding="utf-8")
    assert "page-scripts" in js
    # const/let → var rewrite at column 0
    assert "^(const|let) " in js or "/^(const|let) /gm" in js
    assert "var " in js
    # class → var Name = class Name
    assert "class (" in js or "^class (" in js
    assert "var $1 = class $1" in js
    # error fallback to full navigation during injection
    assert "location.href = url" in js
    assert 'addEventListener("error"' in js or "addEventListener('error'" in js


def test_projects_page_puts_selectTemplate_in_page_scripts(client) -> None:
    r = client.get("/projects")
    assert r.status_code == 200
    html = r.text
    content_pos = html.find('id="content"')
    scripts_pos = html.find('id="page-scripts"')
    assert content_pos != -1
    assert scripts_pos != -1
    assert scripts_pos > content_pos
    # selectTemplate lives in {% block scripts %}, which must render inside page-scripts
    assert "function selectTemplate" in html[scripts_pos:]
    assert "function selectTemplate" not in html[content_pos:scripts_pos]
