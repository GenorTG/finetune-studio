"""Regression: status badges must not imply success/judged when they are not."""

from __future__ import annotations

from pathlib import Path

_CASE_TMPL = (
    Path(__file__).resolve().parents[1]
    / "src"
    / "finetune_studio"
    / "webui"
    / "templates"
    / "_case_results.html"
)
_TRAIN_TMPL = (
    Path(__file__).resolve().parents[1]
    / "src"
    / "finetune_studio"
    / "webui"
    / "templates"
    / "project_training.html"
)
_EXPORT_TMPL = (
    Path(__file__).resolve().parents[1]
    / "src"
    / "finetune_studio"
    / "webui"
    / "templates"
    / "export_models.html"
)
_BENCH_TMPL = (
    Path(__file__).resolve().parents[1]
    / "src"
    / "finetune_studio"
    / "webui"
    / "templates"
    / "benchmarks.html"
)


def test_case_results_hide_pass_rate_when_unjudged() -> None:
    html = _CASE_TMPL.read_text(encoding="utf-8")
    assert "unjudged" in html
    assert "pass_rate: —" in html
    assert "scores.judged" in html
    assert "eval_kind" in html


def test_train_status_badge_idle_is_not_amber() -> None:
    html = _TRAIN_TMPL.read_text(encoding="utf-8")
    # Idle must not look like an in-progress (amber) state.
    assert 'id="train-status-badge"' in html
    assert 'id="train-status-badge">idle</span>' in html or (
        'id="train-status-badge"' in html and "solid-amber" not in html.split('id="train-status-badge"', 1)[1][:80]
    )
    snippet = html.split('id="train-status-badge"', 1)[1][:120]
    assert "solid-amber" not in snippet


def test_export_gptq_export_only_badge_when_optimum_missing() -> None:
    html = _EXPORT_TMPL.read_text(encoding="utf-8")
    assert "export only" in html
    assert "export-gptq-optimum-hint" in html


def test_benchmarks_recent_scores_show_judged_column() -> None:
    html = _BENCH_TMPL.read_text(encoding="utf-8")
    assert "<th>Judged</th>" in html
    assert "Train-set eval" in html
    assert "synthetic/offline" in html
