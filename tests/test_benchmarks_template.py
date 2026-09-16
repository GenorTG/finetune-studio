"""Regression tests for benchmarks.html per-case table (QABUG-012)."""

from __future__ import annotations

from pathlib import Path

from jinja2 import Environment, select_autoescape

_PARTIAL = (
    Path(__file__).resolve().parents[1]
    / "src"
    / "finetune_studio"
    / "webui"
    / "templates"
    / "_case_results.html"
)
_BENCH = (
    Path(__file__).resolve().parents[1]
    / "src"
    / "finetune_studio"
    / "webui"
    / "templates"
    / "benchmarks.html"
)


def _render_cases(cases: list[dict], scores: dict | None = None) -> str:
    body = _PARTIAL.read_text(encoding="utf-8")
    env = Environment(autoescape=select_autoescape(["html", "xml"]))
    return env.from_string(body).render(cases=cases, scores=scores or {})


def test_benchmarks_template_renders_per_case_table() -> None:
    assert '{% include "_case_results.html" %}' in _BENCH.read_text(encoding="utf-8")
    html = _render_cases(
        [
            {
                "name": "pass_case",
                "category": "geo",
                "question": "Capital of France?",
                "correct_answer": "Paris",
                "model_answer": "Paris",
                "verdict": "pass",
                "judge_reasoning": "ok",
                "time_ms": 10,
            },
            {
                "name": "fail_case",
                "category": "geo",
                "question": "Capital of France?",
                "correct_answer": "Paris",
                "model_answer": "London",
                "verdict": "fail",
                "judge_reasoning": "miss",
                "time_ms": 12,
            },
        ]
    )
    assert "pass_case" in html
    assert "fail_case" in html
    assert "verdict-badge verdict-pass" in html
    assert "verdict-badge verdict-fail" in html
    assert 'id="case-results-table"' in html


def test_benchmarks_template_shows_scores_summary() -> None:
    html = _render_cases(
        [],
        scores={
            "total": 6,
            "passed": 4,
            "failed": 2,
            "pass_rate": 0.67,
        },
    )
    assert "case-scores-summary" in html
    assert "judged: 0" in html
    assert "pass_rate: —" in html


def test_case_results_table_is_horizontally_scrollable() -> None:
    """Long question/expected must not clip verdict/time at ~930px viewports."""
    long_q = "Q" * 400
    long_exp = "E" * 400
    html = _render_cases(
        [
            {
                "name": "wide_case",
                "category": "geo",
                "question": long_q,
                "correct_answer": long_exp,
                "model_answer": "Paris",
                "verdict": "pass",
                "judge_reasoning": "ok",
                "time_ms": 42,
            },
        ]
    )
    assert 'class="case-results-scroll"' in html
    assert 'role="region"' in html
    assert 'aria-label="Case results"' in html
    assert "case-col-text" in html
    assert "case-col-verdict" in html
    assert "case-col-time" in html
    assert long_q in html
    assert long_exp in html
    assert "verdict-badge verdict-pass" in html
    assert "42" in html

    css = (
        Path(__file__).resolve().parents[1]
        / "src"
        / "finetune_studio"
        / "webui"
        / "static"
        / "css"
        / "app.css"
    ).read_text(encoding="utf-8")
    assert ".case-results-scroll" in css
    assert "overflow-x: auto" in css
    assert "table-layout: fixed" in css
    assert "overflow-wrap: anywhere" in css
    assert "min-width: 64rem" in css


def test_benchmarks_template_suite_catalog_not_raw_json() -> None:
    body = _BENCH.read_text(encoding="utf-8")
    assert 'id="suite-catalog"' in body
    assert "suite_type" in body
    assert "synthetic smoke" in body.lower() or "synthetic_smoke" in body
    assert "industry ·" not in body
    # Catalog is a table of labels/types — not a dump of suite dicts
    assert "<pre>{{ suites" not in body
    assert "{{ suites|tojson }}" in body  # JS only, not visible catalog
