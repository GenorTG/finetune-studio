"""Settings always offers the in-app updater: Linux is the only supported platform."""

from __future__ import annotations

from pathlib import Path

_TEMPLATE = Path(__file__).resolve().parents[1] / "src" / "finetune_studio" / "webui" / "templates" / "settings.html"


def test_settings_template_always_shows_the_update_controls() -> None:
    template = _TEMPLATE.read_text(encoding="utf-8")
    assert "btn-update-apply" in template and "btn-update-repair" in template
    assert "update_supported" not in template
    assert "Windows" not in template and "install.ps1" not in template and "run.bat" not in template
