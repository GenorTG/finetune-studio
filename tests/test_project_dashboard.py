"""Unit tests for project overview dashboard helpers."""
from __future__ import annotations

from finetune_studio.webui.project_dashboard import (
    build_activity,
    build_dashboard_ctx,
    format_relative,
    models_size_gb,
    recent_models,
    recent_runs,
    run_status_counts,
    truncate_description,
)


def test_format_relative_buckets() -> None:
    now = 1_000_000.0
    assert format_relative(now - 10, now=now) == "just now"
    assert format_relative(now - 120, now=now) == "2m ago"
    assert format_relative(now - 7200, now=now) == "2h ago"
    assert format_relative(None, now=now) == "—"


def test_truncate_description() -> None:
    short = truncate_description("hello")
    assert short["long"] is False
    assert short["short"] == "hello"
    long = truncate_description("x" * 200, limit=120)
    assert long["long"] is True
    assert len(long["short"]) <= 121


def test_run_status_counts_and_recent() -> None:
    runs = [
        {"id": "a", "name": "A", "status": "done", "started_at": 10.0},
        {"id": "b", "name": "B", "status": "running", "started_at": 30.0},
        {"id": "c", "name": "C", "status": "failed", "started_at": 20.0},
        {"id": "d", "name": "D", "status": "stopped", "started_at": 5.0},
        {"id": "e", "name": "E", "status": "done", "started_at": 40.0},
        {"id": "f", "name": "F", "status": "done", "started_at": 50.0},
    ]
    counts = run_status_counts(runs)
    assert counts == {"done": 3, "running": 1, "failed": 1}
    top = recent_runs(runs, limit=5)
    assert [r["id"] for r in top] == ["f", "e", "b", "c", "a"]


def test_recent_models_by_mtime() -> None:
    models = [
        {"name": "old", "mtime": 1.0, "size_gb": 1.5},
        {"name": "new", "mtime": 9.0, "size_gb": 2.0},
        {"name": "mid", "mtime": 5.0, "size_gb": 0.5},
    ]
    top = recent_models(models, limit=2)
    assert [m["name"] for m in top] == ["new", "mid"]
    assert models_size_gb(models) == 4.0


def test_build_activity_and_dashboard_ctx() -> None:
    runs = [
        {
            "id": "r1",
            "name": "run-one",
            "status": "done",
            "started_at": 100.0,
            "finished_at": 200.0,
            "benchmarks": [{"suite_name": "smoke", "ran_at": 250.0}],
        }
    ]
    models = [{"name": "r1/gguf", "mtime": 300.0, "format": "gguf", "size_gb": 1.0}]
    files = [{"original_name": "a.txt", "uploaded_at": 400.0}]
    activity = build_activity(
        runs=runs, models=models, files=files, pid="abc", limit=10, now=500.0
    )
    assert len(activity) >= 4
    assert activity[0]["kind"] == "upload"

    project = {
        "description": "short",
        "runs": runs,
        "models": models,
        "datasets": [{"id": "d1"}],
    }
    ctx = build_dashboard_ctx(project, "abc", files=files, now=500.0)
    assert ctx["stats"]["files"] == 1
    assert ctx["stats"]["datasets"] == 1
    assert ctx["stats"]["runs"] == 1
    assert ctx["stats"]["models"] == 1
    assert ctx["runs_started_max"] == 100.0
    assert ctx["desc_long"] is False
