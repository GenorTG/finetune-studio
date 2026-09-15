"""Regression: benchmarks page clears NO_DATA empties when results exist."""

from __future__ import annotations

from pathlib import Path

_BENCH = (
    Path(__file__).resolve().parents[1]
    / "src"
    / "finetune_studio"
    / "webui"
    / "templates"
    / "benchmarks.html"
)
_CSS = (
    Path(__file__).resolve().parents[1]
    / "src"
    / "finetune_studio"
    / "webui"
    / "static"
    / "css"
    / "app.css"
)


def test_benchmarks_template_clears_empty_chrome_on_compare_results() -> None:
    html = _BENCH.read_text(encoding="utf-8")
    assert "hideCmpEmpty" in html
    assert "showCmpEmpty" in html
    assert 'classList.remove("empty")' in html
    assert "is-filled" in html
    assert 'id="bench-cases-panel"' in html
    assert 'id="bench-cases-empty"' in html
    assert 'id="bench-scores-empty"' in html
    # Accept both done and completed for RUN enablement.
    assert "st in ('done', 'completed')" in html
    assert "Needs a finished run" in html
    # No primary JSON dump of cases/scores.
    assert "<pre>{{ cases" not in html
    assert "cases|tojson(indent" not in html
    assert "all_benchmarks|tojson(indent" not in html


def test_css_strips_no_data_when_empty_hidden() -> None:
    css = _CSS.read_text(encoding="utf-8")
    assert ".empty[hidden]" in css
    assert ".empty.is-filled" in css
    assert "content: none" in css


def test_benchmarks_page_omits_cases_empty_when_results_exist(client) -> None:
    from finetune_studio import db

    pid = client.post(
        "/api/projects", json={"name": "Bench No-Data"}
    ).json()["id"]
    run = db.create_run(
        project_id=pid, name="r1", base_model="/m", data_path="/d"
    )
    db.update_run(run["id"], status="done", output_path="/tmp/out")
    bench = db.create_benchmark(
        run["id"],
        "default",
        {"pass_rate": 80.0, "passed": 4, "failed": 1, "total": 5},
        time_ms=10,
    )
    db.create_case(
        bench["id"],
        run["id"],
        "c1",
        "geo",
        "Capital?",
        "Paris",
        "Paris",
        [],
        verdict="pass",
    )

    body = client.get(f"/projects/{pid}/benchmarks").text
    assert "Paris" in body
    assert 'id="case-results-table"' in body or "case-results-table" in body
    # Stale cases empty must not sit beside loaded results.
    assert 'id="bench-cases-empty"' not in body
    assert "No case results yet" not in body
    # Scores table present; scores empty absent.
    assert 'id="bench-scores-table"' in body
    assert 'id="bench-scores-empty"' not in body
    # Compare placeholder may exist but must not be marked filled yet.
    assert 'id="cmp-empty"' in body
    assert 'data-empty-placeholder="1"' in body


def test_benchmarks_page_enables_run_for_completed_status(client) -> None:
    from finetune_studio import db

    pid = client.post(
        "/api/projects", json={"name": "Bench Completed"}
    ).json()["id"]
    run = db.create_run(
        project_id=pid, name="legacy", base_model="/m", data_path="/d"
    )
    db.update_run(run["id"], status="completed", output_path="/tmp/out")
    body = client.get(f"/projects/{pid}/benchmarks").text
    # Completed run with output_path must not keep RUN disabled for status alone.
    assert "Needs a finished run (status done/completed)" not in body
