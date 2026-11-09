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
        "production_run": "r1",
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
    assert ctx["production_run"] == {"id": "r1", "name": "run-one"}
    assert ctx["setup_checklist"]["items"][0]["state"] == "done"
    assert ctx["setup_checklist"]["items"][1]["state"] == "next"
    assert ctx["setup_checklist"]["next"] == {"label": "Build the Smart Brain", "href": "/brain"}


def test_setup_checklist_tracks_live_project_state() -> None:
    from finetune_studio.webui.project_dashboard import build_setup_checklist

    empty = build_setup_checklist({})
    assert empty["items"][0]["state"] == "next"
    assert empty["next"] == {"label": "Add source files", "href": "/data"}
    assert not empty["complete"]

    built = build_setup_checklist({}, files=[{"id": "f1"}], brain_built=True)
    assert [item["state"] for item in built["items"]] == ["done", "done", "next"]
    assert built["next"] == {"label": "Ask your documents", "href": "/brain"}   # built: use it, the test stays listed
    assert not built["complete"]

    done = build_setup_checklist({}, files=[{"id": "f1"}], brain_built=True, brain_tested=True)
    assert all(item["state"] == "done" for item in done["items"])
    assert done["complete"]
    assert done["next"] == {"label": "Ask your documents", "href": "/brain"}


def test_a_trained_project_without_an_index_still_starts_at_the_smart_brain() -> None:
    """Training no longer counts toward the checklist: it is the optional path."""
    from finetune_studio.webui.project_dashboard import build_setup_checklist

    trained = {"production_run": "r1", "datasets": [{"id": "d1"}], "runs": [{"status": "done", "benchmarks": [{"status": "done"}]}]}
    assert build_setup_checklist(trained, files=[{"id": "f1"}])["next"]["href"] == "/brain"


def test_brain_summary_reads_the_manifest_and_survives_a_missing_one(tmp_path, monkeypatch) -> None:
    import json

    from finetune_studio.webui import project_dashboard as pd

    monkeypatch.setattr("finetune_studio.data.fs.paths.brain_index_dir", lambda pid: tmp_path / pid)
    assert pd.brain_summary("none") == {"built": False, "documents": 0, "chunks": 0}
    (tmp_path / "p1").mkdir()
    (tmp_path / "p1" / "manifest.json").write_text(json.dumps({"documents": 4, "chunks": 56}))
    assert pd.brain_summary("p1") == {"built": True, "documents": 4, "chunks": 56}
    (tmp_path / "p2").mkdir()
    (tmp_path / "p2" / "manifest.json").write_text("{not json")
    assert pd.brain_summary("p2")["built"] is False


def test_testing_quiz_is_a_test_event_not_a_benchmark() -> None:
    runs = [{"id": "r1", "name": "Run r1", "status": "done", "started_at": 1.0,
             "benchmarks": [{"suite_name": "full-ingested-corpus.json", "kind": "suite", "ran_at": 5.0},
                            {"suite_name": "mmlu-smoke", "kind": "public", "ran_at": 6.0}]}]
    msgs = {e["message"]: e["href"] for e in build_activity(runs=runs, models=[], files=[], pid="p1", now=10.0)}
    assert msgs["Test · full-ingested-corpus on Run r1"] == "/projects/p1/testing"
    assert msgs["Benchmark · mmlu-smoke on Run r1"] == "/projects/p1/benchmarks"
