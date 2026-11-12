"""Quick work: "Preset by model size" follows the base model until the user picks a preset.

Live 2026-11-03: Qwen3-0.6B selected, the preset stayed on "Standard — Balanced (3B–9B)".
"""
from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path

import pytest

TEMPLATE = Path(__file__).resolve().parents[1] / "src/finetune_studio/webui/templates/project_wizard.html"


def _html() -> str:
    return TEMPLATE.read_text(encoding="utf-8")


def test_options_carry_their_size_and_the_match_is_wired() -> None:
    html = _html()
    assert 'data-size-gb="{{ model.size_gb }}"' in html
    assert 'modelEl.addEventListener("change", function () { wizMatchPreset(true); });' in html
    assert 'sel.addEventListener("change", function () { wizPresetPicked = true; });' in html
    assert "if (wizPresetPicked || !modelEl || !sel) return;" in html   # a preset the user chose is never overridden
    assert "wizMatchPreset(false);" in html                              # the project's base model on page load


@pytest.mark.parametrize(("label", "size_gb", "billions"), [
    ("Qwen/Qwen3-0.6B (safetensors, 1.4 GB)", "1.4", 0.6),
    ("unsloth/Llama-3.2-3B-Instruct (safetensors, 5.98 GB)", "5.98", 3),
    ("Qwen3.5-9B (safetensors, 17.98 GB)", "17.98", 9),
    ("ibm-granite/granite-3.3-2b-instruct (safetensors, 4.72 GB)", "4.72", 2),
    ("microsoft/Phi-4-mini-instruct (safetensors, 7.15 GB)", "7.15", 3.575),   # no size in the name: bf16 weights
])
def test_model_size_is_read_from_the_name_or_the_weights(label: str, size_gb: str, billions: float) -> None:
    node = shutil.which("node")
    if node is None:
        pytest.skip("Node.js is needed to run the template's size helper")
    html = _html()
    start = html.index("function wizModelBillions")
    src = html[start:html.index("function wizMatchPreset", start)]
    script = (f"{src}\nconst opt = {{textContent: {json.dumps(label)}, getAttribute: () => {json.dumps(size_gb)}}};\n"
              "process.stdout.write(String(wizModelBillions(opt)));")
    out = subprocess.run([node, "-e", script], capture_output=True, text=True, timeout=10, check=True).stdout
    assert float(out) == pytest.approx(billions)
