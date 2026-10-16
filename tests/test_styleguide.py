"""Styleguide route and component contract checks."""

from pathlib import Path

from fastapi.testclient import TestClient

_ROOT = Path(__file__).resolve().parents[1]
_WEB = _ROOT / "src" / "finetune_studio" / "webui"


def test_styleguide_is_reachable_and_renders_shared_components(client: TestClient) -> None:
    response = client.get("/styleguide")

    assert response.status_code == 200
    body = response.text
    assert "Interface styleguide" in body
    for component in (
        'id="sg-actions"',
        'id="sg-fields"',
        'id="sg-status"',
        'id="sg-data"',
        'id="sg-flow"',
        'role="progressbar"',
        'role="alert"',
    ):
        assert component in body
    assert 'href="/static/css/tokens.css?v=1"' in body
    assert 'href="/static/css/app.css?v=98"' in body
    assert 'href="/static/css/typography.css?v=1"' in body


def test_styleguide_samples_follow_existing_theme_and_layout_tokens() -> None:
    css = (_WEB / "static" / "css" / "app.css").read_text(encoding="utf-8")
    typography = (_WEB / "static" / "css" / "typography.css").read_text(encoding="utf-8")
    template = (_WEB / "templates" / "styleguide.html").read_text(encoding="utf-8")
    settings = (_WEB / "templates" / "settings.html").read_text(encoding="utf-8")

    assert "var(--bg-panel)" in css
    assert "var(--text-muted)" in css
    assert "var(--accent)" in css
    assert "#content *" in typography
    assert "#content .mono" in typography
    assert "styleguide-section" in template
    assert "styleguide-steps" in template
    assert 'href="/styleguide"' in settings
