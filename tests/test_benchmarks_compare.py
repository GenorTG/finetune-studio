"""Tests for benchmarks compare tab + GET /api/benchmarks/.../compare."""
from __future__ import annotations

import time


def _project(client) -> str:
    r = client.post("/api/projects", json={"name": "Benchmarks Compare"})
    assert r.status_code in (200, 201), r.text
    return r.json()["id"]


def _run(pid: str, name: str, status: str = "done") -> dict:
    from finetune_studio import db

    run = db.create_run(project_id=pid, name=name, base_model="/m", data_path="/d")
    if status != "queued":
        db.update_run(run["id"], status=status)
        run = db.get_run(run["id"])
    return run  # type: ignore[return-value]


def test_benchmarks_page_renders_both_tabs(client) -> None:
    pid = _project(client)
    _run(pid, "r-a")
    _run(pid, "r-b")
    r = client.get(f"/projects/{pid}/benchmarks")
    assert r.status_code == 200
    body = r.text
    assert 'id="tab-recent"' in body
    assert 'id="tab-compare"' in body
    assert "Recent scores" in body
    assert "Compare two runs" in body
    assert 'data-tab="recent"' in body
    assert 'data-tab="compare"' in body


def test_compare_tab_default_shows_pickers_no_results(client) -> None:
    pid = _project(client)
    _run(pid, "alpha")
    _run(pid, "beta")
    r = client.get(f"/projects/{pid}/benchmarks")
    assert r.status_code == 200
    body = r.text
    assert 'id="cmp-run-a"' in body
    assert 'id="cmp-run-b"' in body
    assert 'id="cmp-run-btn"' in body
    assert "Pick two runs and click Run comparison to see per-suite score diff" in body
    assert 'id="cmp-tbody"' in body
    # Results table wrap hidden until JS fills it
    assert 'id="cmp-table-wrap"' in body
    assert 'style="display:none;"' in body
    assert "runComparison" in body


def test_compare_load_returns_per_suite_shape(client) -> None:
    from finetune_studio import db

    pid = _project(client)
    run_a = _run(pid, "baseline")
    run_b = _run(pid, "candidate")
    db.create_benchmark(
        run_a["id"],
        "default",
        {"pass_rate": 60.0, "passed": 3, "failed": 2, "total": 5},
        time_ms=10,
    )
    db.create_benchmark(
        run_a["id"],
        "tool_calling",
        {"pass_rate": 40.0, "passed": 2, "failed": 3, "total": 5},
        time_ms=10,
    )
    db.create_benchmark(
        run_b["id"],
        "default",
        {"pass_rate": 80.0, "passed": 4, "failed": 1, "total": 5},
        time_ms=10,
    )
    db.create_benchmark(
        run_b["id"],
        "tool_calling",
        {"pass_rate": 50.0, "passed": 2, "failed": 2, "total": 4},
        time_ms=10,
    )

    r = client.get(
        f"/api/benchmarks/projects/{pid}/compare",
        params={"run_a": run_a["id"], "run_b": run_b["id"]},
    )
    assert r.status_code == 200, r.text
    data = r.json()
    assert data["run_a"]["id"] == run_a["id"]
    assert data["run_b"]["id"] == run_b["id"]
    assert "suites" in data
    assert "deltas" in data
    by_suite = {row["suite"]: row for row in data["suites"]}
    assert by_suite["default"]["run_a"] == 60.0
    assert by_suite["default"]["run_b"] == 80.0
    assert by_suite["default"]["delta"] == 20.0
    assert by_suite["tool_calling"]["delta"] == 10.0


def test_compare_load_missing_run_ids_returns_400(client) -> None:
    pid = _project(client)
    r = client.get(f"/api/benchmarks/projects/{pid}/compare")
    assert r.status_code == 400
    assert "run_a" in r.json().get("error", "")

    r2 = client.get(
        f"/api/benchmarks/projects/{pid}/compare",
        params={"run_a": "only-a"},
    )
    assert r2.status_code == 400


def test_compare_load_unknown_run_returns_404(client) -> None:
    pid = _project(client)
    run = _run(pid, "solo")
    r = client.get(
        f"/api/benchmarks/projects/{pid}/compare",
        params={"run_a": run["id"], "run_b": "missing-run-id"},
    )
    assert r.status_code == 404
    assert "not found" in r.json().get("error", "").lower()


def test_compare_defaults_prefer_done_runs(client) -> None:
    """Oldest done → Run A; newest done → Run B."""
    from finetune_studio import db
    from finetune_studio.db.connection import cursor

    pid = _project(client)
    older = _run(pid, "older-done", status="done")
    newer = _run(pid, "newer-done", status="done")
    base = time.time()
    with cursor() as c:
        c.execute(
            "UPDATE training_runs SET created_at = ? WHERE id = ?",
            (base - 100, older["id"]),
        )
        c.execute(
            "UPDATE training_runs SET created_at = ? WHERE id = ?",
            (base, newer["id"]),
        )
    # refresh list order
    assert db.list_runs(pid)[0]["id"] == newer["id"]

    r = client.get(f"/projects/{pid}/benchmarks")
    assert r.status_code == 200
    body = r.text
    # selected on A should be older; on B newer
    assert f'value="{older["id"]}" selected' in body or (
        f'value="{older["id"]}"' in body and "selected" in body
    )
    assert f'id="cmp-run-a"' in body
    # Parse roughly: option for older is selected in cmp-run-a
    a_block = body.split('id="cmp-run-a"')[1].split("</select>")[0]
    b_block = body.split('id="cmp-run-b"')[1].split("</select>")[0]
    a_tail = a_block.split(older["id"])[1][:40]
    b_tail = b_block.split(newer["id"])[1][:40]
    assert f'value="{older["id"]}"' in a_block and "selected" in a_tail
    assert f'value="{newer["id"]}"' in b_block and "selected" in b_tail
