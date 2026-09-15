"""Tests for project testing page suite dropdown."""
from __future__ import annotations

import time


def _project(client) -> str:
    r = client.post("/api/projects", json={"name": "Testing Suite Dropdown"})
    assert r.status_code in (200, 201), r.text
    return r.json()["id"]


def test_testing_page_renders_suite_select(client) -> None:
    pid = _project(client)
    r = client.get(f"/projects/{pid}/testing")
    assert r.status_code == 200
    body = r.text
    assert 'id="t-suite"' in body
    assert "<select id=\"t-suite\"" in body or "<select id='t-suite'" in body
    assert "— pick a suite —" in body
    # Known suites from _discover_suites always present
    assert 'value="data/benchmarks/default.json"' in body
    assert "default.json" in body
    # Free-text path input must be gone
    assert 'placeholder="path/to/suite.json"' not in body
    assert 'type="text"' not in body or 'id="t-suite" type="text"' not in body
    # Live status during long suite runs (polls existing /api/testing/status)
    assert 'id="t-status"' in body
    assert 'id="t-live-log"' in body
    assert "/api/testing/status" in body


def test_testing_page_suite_options_match_discover(client) -> None:
    from finetune_studio.webui.routes.benchmarks import _discover_suites

    pid = _project(client)
    suites = _discover_suites()
    r = client.get(f"/projects/{pid}/testing")
    assert r.status_code == 200
    for s in suites:
        assert f'value="{s["path"]}"' in r.text


def test_recent_suite_runs_helper() -> None:
    from finetune_studio import db
    from finetune_studio.webui.routes.pages import _recent_suite_runs

    proj = db.create_project(name="Recent Suite Runs")
    pid = proj["id"]
    run = db.create_run(project_id=pid, name="r1", base_model="/m", data_path="/d")
    db.create_benchmark(
        run["id"],
        "default",
        {"pass_rate": 80.0, "passed": 4, "failed": 1, "total": 5},
        time_ms=100,
    )
    rows = _recent_suite_runs(pid, limit=5)
    assert len(rows) == 1
    assert rows[0]["suite_name"] == "default"
    assert rows[0]["run_name"] == "r1"
    assert rows[0]["pass_rate"] == 80.0
    assert "ran_at_str" in rows[0]


def test_testing_page_shows_recent_runs(client) -> None:
    from finetune_studio import db

    pid = _project(client)
    run = db.create_run(
        project_id=pid, name="bench-run", base_model="/m", data_path="/d"
    )
    db.create_benchmark(
        run["id"],
        "tool_calling",
        {"pass_rate": 50.0, "passed": 1, "failed": 1, "total": 2},
        time_ms=50,
    )
    r = client.get(f"/projects/{pid}/testing")
    assert r.status_code == 200
    assert "Recent runs" in r.text
    assert "tool_calling" in r.text
    assert "bench-run" in r.text
    assert "50.0%" in r.text


def test_recent_suite_runs_respects_limit() -> None:
    from finetune_studio import db
    from finetune_studio.webui.routes.pages import _recent_suite_runs

    proj = db.create_project(name="Limit Suite Runs")
    pid = proj["id"]
    run = db.create_run(project_id=pid, name="r-limit", base_model="/m", data_path="/d")
    base = time.time()
    for i in range(7):
        b = db.create_benchmark(
            run["id"],
            f"suite_{i}",
            {"pass_rate": float(i), "passed": i, "failed": 0, "total": i or 1},
            time_ms=i,
        )
        # bump ran_at so ordering is deterministic
        from finetune_studio.db.connection import cursor

        with cursor() as c:
            c.execute(
                "UPDATE benchmark_runs SET ran_at = ? WHERE id = ?",
                (base + i, b["id"]),
            )
    rows = _recent_suite_runs(pid, limit=5)
    assert len(rows) == 5
    assert rows[0]["suite_name"] == "suite_6"
    assert rows[-1]["suite_name"] == "suite_2"
