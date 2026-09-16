"""Dashboard index template: live resources, no fake static hero stats."""

from __future__ import annotations

from pathlib import Path

_INDEX = (
    Path(__file__).resolve().parents[1]
    / "src"
    / "finetune_studio"
    / "webui"
    / "templates"
    / "index.html"
)
_RESOURCES = (
    Path(__file__).resolve().parents[1]
    / "src"
    / "finetune_studio"
    / "webui"
    / "templates"
    / "_resources.html"
)


def test_dashboard_hero_has_no_fake_static_stats() -> None:
    body = _INDEX.read_text(encoding="utf-8")
    assert "14 nodes connected" not in body
    assert "last scan 00:00:42" not in body
    assert "██▒▒▒▒▒ 28%" not in body
    assert "██████▒▒ 72%" not in body
    # Fake CPU/MEM labels in the hero are gone (stat tile "GPU" elsewhere is ok).
    assert 'class="mono text-xs muted">CPU</div>' not in body
    assert 'class="mono text-xs muted mt-2">MEM</div>' not in body


def test_dashboard_includes_shared_resources_partial() -> None:
    body = _INDEX.read_text(encoding="utf-8")
    assert '{% include "_resources.html" %}' in body
    resources = _RESOURCES.read_text(encoding="utf-8")
    assert "/api/system/resources" in resources
    assert 'id="ram-text"' in resources
    assert 'id="vram-rows"' in resources
    assert 'id="sys-resources"' in resources
