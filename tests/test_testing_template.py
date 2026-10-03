"""Regression tests for the shared per-case results table (QABUG-013).

``_case_results.html`` is included by the live benchmarks page.
"""

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
_BENCHMARKS = (
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


def test_testing_template_renders_per_case_table() -> None:
    src = _BENCHMARKS.read_text(encoding="utf-8")
    assert '{% include "_case_results.html" %}' in src
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


def test_testing_template_shows_scores_summary() -> None:
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
