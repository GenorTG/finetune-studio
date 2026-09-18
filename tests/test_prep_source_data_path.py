"""Regression: prep-run source manifest must keep a resolvable data_path.

Bug 2026-09-18: DataPrepRunner rewrote the QA source row after chunking
without ``data_path``/``path``, so re-running prep from the source picker
("Start prep") failed with "source file missing" after any restart.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from finetune_studio.data import project_filesystem as pfs


@pytest.fixture()
def _fake_backend(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> list[dict[str, Any]]:
    """Fake generator: one deterministic Q&A JSON block, no model needed."""
    calls: list[dict[str, Any]] = []

    def _fake_chat(messages: list[dict[str, str]], max_tokens: int = 1,
                   **_: Any) -> str:
        calls.append({"messages": messages, "max_tokens": max_tokens})
        return (
            '[{"question": "Q1?", "answer": "A1"},'
            ' {"question": "Q2?", "answer": "A2"}]'
        )

    monkeypatch.setattr(
        "finetune_studio.data.prep.runner.resolve_generator",
        lambda: _fake_chat,
        raising=False,
    )
    return calls


def test_source_manifest_keeps_data_path(
    client: Any, _fake_backend: list[dict[str, Any]]
) -> None:
    """After a prep run, the source row must have a data_path that exists."""
    pid = client.post("/api/projects", json={"name": "PathKeeper"}).json()["id"]
    data = b"# Lore\n\nThe disc floats above the Void Sea. Forty weavers sealed the shafts."
    r = client.post(
        f"/api/projects/{pid}/data-prep/upload",
        files={"file": ("lore.txt", data, "text/plain")},
        data={"qa_per_chunk": "2", "difficulty": "medium", "style": "direct"},
    )
    assert r.status_code == 200, r.text
    src = pfs.read_qa_source(pid, pfs.list_qa_sources(pid)[0]["id"])
    dp = src.get("data_path") or src.get("path")
    assert dp, f"source row lost data_path: {src}"
    assert Path(dp).is_file(), f"data_path does not exist on disk: {dp}"
    # The path must point at the content-addressed raw file for this sha.
    assert src["sha256"][:12] in dp
