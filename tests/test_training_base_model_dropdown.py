"""Regression E2E-19: training page must keep the server-rendered model list.

The async /api/hf/local refresh used to `sel.innerHTML = ''` and rebuild the
dropdown from HF downloads only — dropping every other local model and the
project's preselected base_model.
"""
from __future__ import annotations

from pathlib import Path

_TEMPLATE = (
    Path(__file__).resolve().parents[1]
    / "src"
    / "finetune_studio"
    / "webui"
    / "templates"
    / "project_training.html"
)


def _refresh_block() -> str:
    src = _TEMPLATE.read_text(encoding="utf-8")
    return src.split("const sel = document.querySelector('select[name=\"model_path\"]');", 1)[1].split("</script>", 1)[0]


def test_hf_refresh_does_not_wipe_options() -> None:
    assert "sel.innerHTML = ''" not in _refresh_block()


def test_hf_refresh_skips_known_paths() -> None:
    block = _refresh_block()
    assert "known.has(m.path)" in block


def test_server_list_preselects_project_base_model() -> None:
    src = _TEMPLATE.read_text(encoding="utf-8")
    assert "model.path == project.base_model" in src
