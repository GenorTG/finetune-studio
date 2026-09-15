"""Unit tests for Testing model resolution helpers."""

from __future__ import annotations

from finetune_studio.webui.testing_models import (
    default_model_path_for_testing,
    is_run_done,
    models_for_testing_page,
    resolve_latest_merged_model,
)


def test_is_run_done_accepts_done_and_completed() -> None:
    assert is_run_done({"status": "done"}) is True
    assert is_run_done({"status": "completed"}) is True
    assert is_run_done({"status": "running"}) is False
    assert is_run_done({"status": "failed"}) is False


def test_resolve_latest_merged_prefers_ready_merged(
    tmp_path,
) -> None:
    out = tmp_path / "r1"
    merged = out / "merged"
    merged.mkdir(parents=True)
    (merged / "w.safetensors").write_bytes(b"x")

    runs = [
        {"status": "done", "output_path": str(tmp_path / "empty")},
        {"status": "done", "output_path": str(out)},
        {"status": "running", "output_path": str(out)},
    ]
    (tmp_path / "empty").mkdir()
    path = resolve_latest_merged_model(
        "p", list_runs_fn=lambda _pid: runs
    )
    assert path == str(merged)


def test_testing_models_filters_and_defaults() -> None:
    models = [
        {
            "name": "a/gguf",
            "format": "gguf",
            "path": "/p/gguf",
            "mtime": 1.0,
            "size_gb": 0.1,
        },
        {
            "name": "a/merged",
            "format": "safetensors",
            "path": "/p/merged",
            "mtime": 2.0,
            "size_gb": 1.0,
        },
        {
            "name": "a/gptq",
            "format": "gptq",
            "path": "/p/gptq",
            "mtime": 3.0,
            "size_gb": 0.5,
        },
    ]
    usable = models_for_testing_page(models)
    assert [m["path"] for m in usable] == ["/p/merged", "/p/gguf"]
    assert default_model_path_for_testing(usable) == "/p/merged"
