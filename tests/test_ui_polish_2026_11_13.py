"""Walkthrough 2026-11-13 minors: /package URL, labelled controls, blocked Build says why, honest download hint,
a skip link, pills on mobile (m1, m21, m23, m29, m30, p7)."""
from __future__ import annotations

import re
from pathlib import Path

TEMPLATES = Path(__file__).resolve().parents[1] / "src/finetune_studio/webui/templates"
CSS = Path(__file__).resolve().parents[1] / "src/finetune_studio/webui/static/css/app.css"


def test_package_url_and_guide_alias(client) -> None:
    pid = client.post("/api/projects", json={"name": "agent-test-polish"}).json()["id"]
    r = client.get(f"/projects/{pid}/package?x=1", follow_redirects=False)
    assert r.status_code == 307 and r.headers["location"] == f"/projects/{pid}/export?x=1"
    from finetune_studio.guide import registry
    assert registry.PAGE_ALIASES["package"] == "export"
    status = client.get(f"/api/projects/{pid}/brain").json()
    assert isinstance(status["models_on_disk"], bool)


def test_every_walkthrough_control_has_an_accessible_name() -> None:
    ids = {"projects.html": ["proj-search"], "hf_models.html": ["hf-q", "hf-task", "hf-sort"],
           "settings.html": ["access-pw-current", "access-pw-new", "access-token-name"],
           "project_data.html": ["fb-search"], "data_prep.html": ["fl-search"], "brain.html": ["bundle-name", "bundle-pass"],
           "export_models.html": ["mcp-fmt", "mcp-port", "mcp-pass", "mcp-device", "mcp-host"],
           "chat_v2.html": ["chat-inline-model", "chat-input"], "project_wizard.html": ["wiz-version-label"]}
    for name, wanted in ids.items():
        html = (TEMPLATES / name).read_text(encoding="utf-8")
        for i in wanted:
            tag = re.search(r'<(?:input|select|textarea)[^>]*id="%s"[^>]*>' % i, html).group(0)
            assert "aria-label=" in tag or f'for="{i}"' in html, (name, i)


def test_blocked_build_and_download_hint_and_skip_link() -> None:
    export = (TEMPLATES / "export_models.html").read_text(encoding="utf-8")
    assert "window.PACKAGE_BLOCKED = stop && /superseded/i.test" in export
    assert "g('mcp-pkg-btn').disabled = !found.length || !!window.PACKAGE_BLOCKED;" in export
    brain = (TEMPLATES / "brain.html").read_text(encoding="utf-8")
    assert "d.models_on_disk === false ? ' The first build also downloads" in brain
    base = (TEMPLATES / "base.html").read_text(encoding="utf-8")
    assert '<a class="skip-link" href="#content">Skip to content</a>' in base and 'id="content"' in base
    css = CSS.read_text(encoding="utf-8")
    assert ".skip-link:focus" in css and ".card-head > .pill, .card-head > span > .pill { align-self: flex-start" in css
