"""Regression tests for sticky project breadcrumb (QABUG-009)."""

from __future__ import annotations

from pathlib import Path

from jinja2 import Environment, FileSystemLoader, select_autoescape

_TEMPLATES = (
    Path(__file__).resolve().parents[1]
    / "src"
    / "finetune_studio"
    / "webui"
    / "templates"
)


def _render_base(**kwargs: object) -> str:
    env = Environment(
        loader=FileSystemLoader(str(_TEMPLATES)),
        autoescape=select_autoescape(["html", "xml"]),
    )
    child = env.from_string(
        '{% extends "base.html" %}'
        "{% block content %}crumb-probe{% endblock %}"
    )
    return child.render(app_version="0.0-test", request=None, **kwargs)


def test_breadcrumb_present_when_pid_defined() -> None:
    html = _render_base(pid="abc123", project={"name": "nightly-qa", "id": "abc123"})
    assert 'id="project-breadcrumb"' in html
    assert 'href="/projects/abc123"' in html
    assert "nightly-qa" in html


def test_breadcrumb_absent_when_no_pid() -> None:
    html = _render_base()
    # No pid → breadcrumb block must not render.
    assert 'id="project-breadcrumb"' not in html
    assert html.count("project-breadcrumb") == 0
