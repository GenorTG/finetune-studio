"""The Windows Settings page must not offer the Unix-only self-updater."""

from __future__ import annotations

import asyncio
import platform
from pathlib import Path

from finetune_studio.webui.routes import pages

_TEMPLATE = (
    Path(__file__).resolve().parents[1]
    / "src"
    / "finetune_studio"
    / "webui"
    / "templates"
    / "settings.html"
)


def test_settings_route_marks_windows_updater_unsupported(monkeypatch) -> None:
    captured: dict = {}

    def capture_template_response(request, template_name, context):
        captured.update(context)
        return context

    monkeypatch.setattr(pages, "platform", platform, raising=False)
    monkeypatch.setattr(platform, "system", lambda: "Windows")
    monkeypatch.setattr(pages.templates, "TemplateResponse", capture_template_response)

    asyncio.run(pages.settings_page(request=None))

    assert captured["update_supported"] is False


def test_settings_template_shows_manual_windows_update_instead_of_controls() -> None:
    template = _TEMPLATE.read_text(encoding="utf-8")

    assert "{% if update_supported %}" in template
    assert "btn-update-apply" in template
    assert "Update this Windows checkout from PowerShell." in template
    assert "In-app updates and repair are not available on Windows." in template
    assert "install.ps1" in template
    assert "run.bat" in template

    script = (
        _TEMPLATE.parents[1] / "static" / "js" / "settings.js"
    ).read_text(encoding="utf-8")
    assert "if (!document.getElementById('update-status'))" in script
    assert "await loadDebug();\n    return;" in script
