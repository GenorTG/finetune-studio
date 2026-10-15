"""The onboarding tour points at things that exist and says true things.

Rewritten 2026-10-03: the old tour toured chrome (session bar, pages) instead
of the workflow, claimed models cache in ~/.cache, used the retired "HF
EXPLORER" name, ignored its own ``align`` field (dialog always covered the
page centre) and restarted from step 1 after any full page reload.
"""

from __future__ import annotations

import re
from pathlib import Path

from finetune_studio import webui

_WEBUI = Path(webui.__file__).parent
_JS = (_WEBUI / "static" / "js" / "tutorial.js").read_text(encoding="utf-8")
_TEMPLATES = "".join(p.read_text(encoding="utf-8") for p in (_WEBUI / "templates").glob("*.html"))


def test_every_highlight_target_exists_in_the_templates() -> None:
    targets = re.findall(r'target:\s*"([^"]+)"', _JS)
    assert targets, "tour has no highlighted steps"
    for sel in targets:
        if sel.startswith("#"):
            assert f'id="{sel[1:]}"' in _TEMPLATES, sel
        else:
            assert sel.lstrip(".") in _TEMPLATES, sel


def test_every_api_the_tour_calls_exists() -> None:
    from finetune_studio.webui.app import app

    paths = set(app.openapi()["paths"])
    called = {u.split("?")[0] for u in re.findall(r'getJSON\("([^"]+)"\)', _JS)}
    assert called, "tour reads no live state"
    assert called <= paths, called - paths


def test_tour_covers_the_workflow_and_drops_stale_claims() -> None:
    for must in ("GET A BASE MODEL", "CREATE A PROJECT", "QUICK WORK", "CHECK THE RESULT",
                 "/models/explore", "/projects#new"):
        assert must in _JS, must
    for stale in ("~/.cache", "HF EXPLORER", "qa3 rag", "HACKER"):
        assert stale not in _JS, stale


def test_tour_anchors_dialog_and_resumes_after_reload() -> None:
    assert "function place(" in _JS and 'classList.add("anchored")' in _JS
    assert "sessionStorage" in _JS and "readStep()" in _JS
    css = (_WEBUI / "static" / "css" / "app.css").read_text(encoding="utf-8")
    assert ".tutorial-dialog.anchored" in css and ".tutorial-live" in css


def test_projects_page_opens_new_form_from_hash() -> None:
    html = (_WEBUI / "templates" / "projects.html").read_text(encoding="utf-8")
    assert 'id="new-project-btn"' in html and "location.hash !== '#new'" in html
    assert "'fts:navigated', openIfNew" in html


def test_page_actions_block_is_rendered() -> None:
    """base.html must render each page's topbar_right block. The session-bar
    redesign (b960ca3) dropped it, leaving Import project with no button."""
    from jinja2 import Environment, FileSystemLoader

    env = Environment(loader=FileSystemLoader(str(_WEBUI / "templates")))
    child = env.from_string(
        '{% extends "base.html" %}{% block topbar_right %}<button id="probe-action">x</button>'
        '{% endblock %}{% block content %}<p>c</p>{% endblock %}'
    )
    html = child.render(request=None, projects=[], app_version="t")
    assert 'id="page-actions"' in html and 'id="probe-action"' in html
    plain = env.from_string('{% extends "base.html" %}{% block content %}<p>c</p>{% endblock %}')
    assert 'id="page-actions"' not in plain.render(request=None, projects=[], app_version="t")


def test_spa_fallback_reloads_same_page_hash_urls() -> None:
    """A failed SPA swap falls back to a full load; for /same/path#hash a bare
    location.href assignment only scrolls, leaving the page half-swapped."""
    spa = (_WEBUI / "static" / "js" / "spa.js").read_text(encoding="utf-8")
    assert "function hardNavigate(" in spa and "location.reload()" in spa
    assert "location.href = url;" not in spa
