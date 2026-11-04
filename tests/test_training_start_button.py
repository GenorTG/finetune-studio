"""The Training page's Start button: busy feedback comes from the global form handler, and it stays off during a run.

Found by a live training flow: the page set "Starting…" before app.js saved the button's label, so app.js restored
"Starting…" when the request ended and the button said so until a reload.
"""
from __future__ import annotations

from pathlib import Path

_WEB = Path(__file__).resolve().parents[1] / "src" / "finetune_studio" / "webui"
_PAGE = (_WEB / "templates" / "project_training.html").read_text(encoding="utf-8")
_APP = (_WEB / "static" / "js" / "app.js").read_text(encoding="utf-8")


def test_page_never_writes_the_start_label_itself() -> None:
    assert 'startBtn.textContent' not in _PAGE
    assert 'data-api="/api/training/start" data-busy="Starting…"' in _PAGE


def test_global_handler_restores_the_saved_label() -> None:
    assert 'form.dataset.busy || "Working…"' in _APP
    assert "btn.innerHTML = btnHtml" in _APP


def test_start_is_off_while_a_run_is_active() -> None:
    assert "startBtn.disabled = !hasData || runActive;" in _PAGE
    assert "runActive = !!ACTIVE[st];" in _PAGE
    assert "A training run is in progress" in _PAGE
