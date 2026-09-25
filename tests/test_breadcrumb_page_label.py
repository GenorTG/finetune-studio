"""Breadcrumb tab label must match the current project page; SPA must swap it."""

from __future__ import annotations

from pathlib import Path

from jinja2 import Environment, FileSystemLoader, select_autoescape

_ROOT = Path(__file__).resolve().parents[1]
_TEMPLATES = _ROOT / "src" / "finetune_studio" / "webui" / "templates"
_SPA_JS = _ROOT / "src" / "finetune_studio" / "webui" / "static" / "js" / "spa.js"

_PROJECT_TEMPLATES = {
    "project.html": "overview",
    "project_training.html": "training",
    "data_prep.html": "pairs",
    "project_testing.html": "testing",
    "rag.html": "rag",
    "chat_v2.html": "chat",
    "export_models.html": "export",
    "project_data.html": "data",
    "project_models.html": "models",
    "project_settings.html": "settings",
    "benchmarks.html": "benchmarks",
}


def test_project_templates_set_breadcrumb_tab() -> None:
    for name, expected in _PROJECT_TEMPLATES.items():
        text = (_TEMPLATES / name).read_text(encoding="utf-8")
        assert (
            "{% block breadcrumb_tab %}" + expected + "{% endblock %}"
        ) in text, f"{name} missing breadcrumb_tab={expected}"


def test_rendered_training_breadcrumb_not_overview() -> None:
    env = Environment(
        loader=FileSystemLoader(str(_TEMPLATES)),
        autoescape=select_autoescape(["html", "xml"]),
    )
    child = env.from_string(
        '{% extends "base.html" %}'
        "{% block breadcrumb_tab %}training{% endblock %}"
        "{% block content %}ok{% endblock %}"
    )
    html = child.render(
        app_version="0.0-test",
        request=None,
        pid="p1",
        project={"id": "p1", "name": "demo"},
    )
    assert 'id="project-breadcrumb"' in html
    assert '<span class="bc-tab">training</span>' in html
    assert '<span class="bc-tab">overview</span>' not in html


def test_spa_swaps_breadcrumb_on_navigate() -> None:
    src = _SPA_JS.read_text(encoding="utf-8")
    assert "project-breadcrumb" in src
    assert "breadcrumbHTML" in src
    assert "breadcrumbPresent" in src
    # Must create crumb when entering a project (not only update innerHTML).
    assert "breadcrumbOuter" in src
    assert "syncShellNav" in src


def test_no_updated_every_2s_user_facing() -> None:
    """SSE surfaces must not claim a 2s poll cadence."""
    base = (_TEMPLATES / "base.html").read_text(encoding="utf-8")
    activity = (
        _ROOT / "src/finetune_studio/webui/static/js/activity.js"
    ).read_text(encoding="utf-8")
    training = (_TEMPLATES / "project_training.html").read_text(encoding="utf-8")
    for blob in (base, activity, training):
        assert "Updated every 2s" not in blob
    assert "Live</span>" in base
    assert "Streaming while training" in training
    # activity.js must not reintroduce a 2s full redraw.
    assert "setInterval(refresh, 2000)" not in activity
    assert "every 2s" not in activity.lower()