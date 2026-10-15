"""Unit tests for project file-browser helpers."""
from __future__ import annotations

from finetune_studio.webui.project_data_browser import (
    build_file_browser_ctx,
    file_browser_stats,
    format_bytes,
    runs_started_max,
)


def test_format_bytes() -> None:
    assert format_bytes(500) == "500 B"
    assert format_bytes(2048).endswith("KB")
    assert format_bytes(5 * 1024 * 1024).endswith("MB")
    assert format_bytes(None) == "0 B"


def test_file_browser_stats() -> None:
    files = [
        {"size_bytes": 100, "deleted_at": None},
        {"size_bytes": 200, "deleted_at": None},
        {"size_bytes": 50, "deleted_at": 1.0},
    ]
    stats = file_browser_stats(files)
    assert stats["library"] == 2
    assert stats["trash"] == 1
    assert stats["total"] == 3
    assert stats["storage_bytes"] == 350


def test_runs_started_max() -> None:
    assert runs_started_max([]) is None
    assert runs_started_max([{"started_at": 10}, {"started_at": 30}]) == 30.0


def test_build_file_browser_ctx() -> None:
    project = {
        "description": "short desc",
        "runs": [{"started_at": 42.0}],
    }
    files = [{"size_bytes": 10, "deleted_at": None}]
    ctx = build_file_browser_ctx(project, files=files)
    assert ctx["desc_short"] == "short desc"
    assert ctx["desc_long"] is False
    assert ctx["fb_stats"]["library"] == 1
    assert ctx["runs_started_max"] == 42.0
