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


def test_local_model_missing_only_flags_absent_local_paths(tmp_path) -> None:
    from finetune_studio.webui.testing_models import local_model_missing

    present = tmp_path / "m.gguf"
    present.write_bytes(b"x")
    assert local_model_missing(str(present)) is False
    assert local_model_missing(str(tmp_path / "gone.gguf")) is True
    assert local_model_missing("Qwen/Qwen3.5-9B") is False  # Hub id, not a local path
    assert local_model_missing("") is False


def test_target_model_refuses_a_folder_no_loader_can_open(tmp_path) -> None:
    import pytest

    from finetune_studio.webui.testing_models import target_model, unloadable_model_dir

    run_dir = tmp_path / "runs" / "abc"
    (run_dir / "adapter").mkdir(parents=True)
    (run_dir / "checkpoint-10").mkdir()
    with pytest.raises(ValueError, match="not a model folder"):
        target_model("pid", str(run_dir))
    merged = tmp_path / "merged"
    merged.mkdir()
    (merged / "config.json").write_text("{}")
    assert target_model("pid", str(merged)) == str(merged)
    (run_dir / "adapter" / "adapter_config.json").write_text("{}")
    assert not unloadable_model_dir(str(run_dir / "adapter"))
    gguf_dir = tmp_path / "gguf"
    gguf_dir.mkdir()
    (gguf_dir / "model-q4_k_m.gguf").write_bytes(b"x")
    assert not unloadable_model_dir(str(gguf_dir))
    assert not unloadable_model_dir("Qwen/Qwen3.5-9B")
