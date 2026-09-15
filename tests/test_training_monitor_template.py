"""Static checks for project_training.html monitor + past-run action guards (E2E-30/31)."""
from __future__ import annotations

from pathlib import Path

TEMPLATE = (
    Path(__file__).resolve().parents[1]
    / "src"
    / "finetune_studio"
    / "webui"
    / "templates"
    / "project_training.html"
)


def _html() -> str:
    return TEMPLATE.read_text(encoding="utf-8")


def test_monitor_polls_training_status_and_updates_bar_log() -> None:
    html = _html()
    assert 'id="train-progress-bar"' in html
    assert 'id="train-log"' in html
    assert "/api/training/status" in html
    assert "train-progress-bar" in html
    assert "log_lines" in html or "s.log_lines" in html
    assert "Step " in html and "loss" in html
    assert "refreshPastRuns" in html
    # Must not rely only on status-text for the live bar/log.
    assert "data-poll=\"/api/training/status-text\"" not in html.split("Live status")[1].split("Past runs")[0]


def test_past_run_actions_require_output_path_and_done() -> None:
    html = _html()
    assert "can_act = run.status == 'done' and run.output_path" in html
    assert "canAct = status === \"done\" && !!out" in html
    assert "Only finished runs with an output path" in html
    assert "No finished output path" in html
    # Failed rows surface error text with title for full message.
    assert "run.error" in html
    assert 'title="{{ run.error }}"' in html or "title=\"{{ run.error }}\"" in html
