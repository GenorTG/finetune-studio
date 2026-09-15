"""Regression: Export UI reflects host GGUF/GPTQ converter availability."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

from finetune_studio.training.export_capabilities import (
    GPTQ_CONVERTER_MISSING_MSG,
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
    assert set(data) >= {
        "gguf",
        "gptq",
        "gguf_hint",
        "gptq_hint",
        "gguf_script",
        "optimum",
        "gptq_inference_hf",
    }
    assert isinstance(data["gguf"], bool)
    assert isinstance(data["gptq"], bool)
    assert isinstance(data["optimum"], bool)
    if not data["gguf"]:
        assert "convert_hf_to_gguf" in data["gguf_hint"]
    if not data["gptq"]:
        assert "gptqmodel" in data["gptq_hint"] or "auto_gptq" in data["gptq_hint"]


def test_export_template_surfaces_optimum_hint() -> None:
    html = _EXPORT_TMPL.read_text(encoding="utf-8")
    assert "export-gptq-optimum-hint" in html
    assert "export only" in html
    assert "optimum" in html.lower()
    assert '.[gptq]' in html or "optimum" in html


def test_export_template_disables_unavailable_formats() -> None:
    html = _EXPORT_TMPL.read_text(encoding="utf-8")
    assert 'id="export-caps"' in html
    assert "export-gguf-unavailable" in html
    assert "export-gptq-unavailable" in html
    assert "checked:not(:disabled)" in html
    assert "data-gguf" in html
    assert "data-gptq" in html
    assert "unavailable" in html
    # Always offer merged as a truthful fallback.
    assert 'value="merged"' in html


def test_export_page_marks_gguf_gptq_unavailable(client) -> None:
    missing = ExportCapabilities(
        gguf=False,
        gptq=False,
        gguf_script=None,
        gguf_hint=GGUF_CONVERTER_MISSING_MSG,
        gptq_hint=GPTQ_CONVERTER_MISSING_MSG,
        optimum=False,
        gptq_inference_hf=False,
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
    assert 'id="export-gptq-unavailable"' in body
    assert "convert_hf_to_gguf" in body
    assert "gptqmodel" in body or "auto_gptq" in body or "auto-gptq" in body
    assert 'value="gguf"' in body
    assert "disabled" in body
    # GGUF must not be the default checked choice when unavailable.
    assert 'value="gguf" checked' not in body.replace("\n", " ")


def test_export_page_enables_gguf_when_converter_present(client) -> None:
    available = ExportCapabilities(
        gguf=True,
        gptq=True,
        gguf_script="/opt/llama.cpp/convert_hf_to_gguf.py",
        gguf_hint="",
        gptq_hint="",
        optimum=True,
        gptq_inference_hf=True,
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
    assert 'id="export-gptq-unavailable"' not in body
    assert 'data-gguf="1"' in body
    assert 'data-gptq="1"' in body
