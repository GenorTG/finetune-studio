"""Regression: Export UI reflects host GGUF converter availability."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

from finetune_studio.training.export_capabilities import (
    ExportCapabilities,
    probe_export_capabilities,
)
from finetune_studio.training.run_export import GGUF_CONVERTER_MISSING_MSG

_EXPORT_TMPL = (
    Path(__file__).resolve().parents[1]
    / "src"
    / "finetune_studio"
    / "webui"
    / "templates"
    / "export_models.html"
)


def test_probe_export_capabilities_shape() -> None:
    caps = probe_export_capabilities()
    assert isinstance(caps, ExportCapabilities)
    data = caps.as_dict()
    assert set(data) >= {"gguf", "gguf_hint", "gguf_script"}
    assert isinstance(data["gguf"], bool)
    if not data["gguf"]:
        assert "convert_hf_to_gguf" in data["gguf_hint"]


def test_export_template_disables_unavailable_formats() -> None:
    html = _EXPORT_TMPL.read_text(encoding="utf-8")
    assert 'id="export-caps"' in html
    assert "export-gguf-unavailable" in html
    assert "checked:not(:disabled)" in html
    assert "data-gguf" in html
    assert "unavailable" in html
    # Always offer merged as a truthful fallback.
    assert 'value="merged"' in html


def test_export_page_marks_gguf_unavailable(client) -> None:
    missing = ExportCapabilities(
        gguf=False,
        gguf_script=None,
        gguf_hint=GGUF_CONVERTER_MISSING_MSG,
    )
    pid = client.post(
        "/api/projects", json={"name": "Caps Unavailable"}
    ).json()["id"]
    with patch(
        "finetune_studio.training.export_capabilities.probe_export_capabilities",
        return_value=missing,
    ):
        body = client.get(f"/projects/{pid}/export").text
    assert 'id="export-gguf-unavailable"' in body
    assert "convert_hf_to_gguf" in body
    assert 'value="gguf"' in body
    assert "disabled" in body
    # GGUF must not be the default checked choice when unavailable.
    assert 'value="gguf" checked' not in body.replace("\n", " ")


def test_export_page_enables_gguf_when_converter_present(client) -> None:
    available = ExportCapabilities(
        gguf=True,
        gguf_script="/opt/llama.cpp/convert_hf_to_gguf.py",
        gguf_hint="",
    )
    pid = client.post(
        "/api/projects", json={"name": "Caps Available"}
    ).json()["id"]
    with patch(
        "finetune_studio.training.export_capabilities.probe_export_capabilities",
        return_value=available,
    ):
        body = client.get(f"/projects/{pid}/export").text
    assert 'id="export-gguf-unavailable"' not in body
    assert 'data-gguf="1"' in body
