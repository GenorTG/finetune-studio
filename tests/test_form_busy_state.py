"""Shared ``form[data-api]`` submit: a long request shows it is working and cannot be submitted twice.

A benchmark RUN answers every question before its POST returns (8 s for 6 smoke cases, minutes for 50 official ones);
the button stayed clickable with nothing on screen, and a second click started a second run that loaded the model again.
"""
from __future__ import annotations

from pathlib import Path

_ROOT = Path(__file__).parents[1] / "src/finetune_studio/webui"


def _submit_handler() -> str:
    js = (_ROOT / "static/js/app.js").read_text(encoding="utf-8")
    start = js.index('form.addEventListener("submit"')
    return js[start:js.index("// Buttons — use event delegation", start)]


def test_submit_refuses_a_second_submit_while_busy() -> None:
    handler = _submit_handler()
    assert 'if (form.dataset.ftsBusy === "1") return;' in handler
    assert 'form.dataset.ftsBusy = "1";' in handler
    assert "delete form.dataset.ftsBusy;" in handler


def test_submit_button_shows_progress_and_is_restored() -> None:
    handler = _submit_handler()
    assert "btn.disabled = true;" in handler
    assert 'form.dataset.busy || "Working…"' in handler
    assert "btn.innerHTML = btnHtml" in handler
    # Both the JSON and the multipart path release the form when the request settles, success or failure.
    assert handler.count(".finally(done)") == 2


def test_benchmark_forms_name_what_is_running() -> None:
    html = (_ROOT / "templates/benchmarks.html").read_text(encoding="utf-8")
    assert html.count('data-busy="Running the benchmark…"') == 2  # each run's RUN and the base-model RUN
    assert 'data-busy="Evaluating…"' in html
