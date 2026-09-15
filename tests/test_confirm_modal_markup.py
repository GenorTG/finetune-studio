"""HF Pull confirm modal must render a clickable OK that resolves true.

Browser-confirmed defect: missing closing quote on class= left the OK button as
``<button class="btn primary data-action=" ok"="">OK</button>``, so the confirm
handler never attached and /api/hf/download was never enqueued.
"""
from __future__ import annotations

from pathlib import Path

_APP_JS = (
    Path(__file__).resolve().parents[1]
    / "src"
    / "finetune_studio"
    / "webui"
    / "static"
    / "js"
    / "app.js"
)


def _confirm_dialog_src() -> str:
    src = _APP_JS.read_text(encoding="utf-8")
    start = src.find("function confirmDialog")
    assert start != -1, "confirmDialog missing"
    end = src.find("function _errorMessage", start)
    assert end != -1
    return src[start:end]


def test_confirm_ok_button_closes_class_attribute() -> None:
    block = _confirm_dialog_src()
    # Exact safe markup after the class-quote fix.
    assert 'class="btn ${variant}" data-fts-modal="ok"' in block
    # Must not leave class open into a data-action attribute (the live bug).
    assert 'data-action="ok"' not in block


def test_confirm_modal_does_not_use_data_action() -> None:
    """Modal chrome must not collide with global [data-action] API delegation."""
    block = _confirm_dialog_src()
    assert "data-fts-modal" in block
    assert 'data-action="cancel"' not in block
    assert 'data-action="ok"' not in block


def test_data_action_listener_skips_modal_overlay() -> None:
    src = _APP_JS.read_text(encoding="utf-8")
    assert 'closest(".modal-overlay")' in src
