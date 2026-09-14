"""API + page smoke tests for the project-scoped data editor."""
from __future__ import annotations

import json
import tempfile
from pathlib import Path


def _make_dataset(rows: list[dict]) -> str:
    """Write a temp JSONL file; return absolute path (API accepts absolute)."""
    fd, path = tempfile.mkstemp(suffix=".jsonl")
    Path(path).write_text(
        "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows),
        encoding="utf-8",
    )
    import os
    os.close(fd)
    return path


def _project(client) -> str:
    r = client.post("/api/projects", json={"name": "Data Editor Test"})
    assert r.status_code in (200, 201), r.text
    return r.json()["id"]


def test_data_editor_page_renders(client) -> None:
    pid = _project(client)
    r = client.get(f"/projects/{pid}/data/demo/suite.json")
    assert r.status_code == 200
    body = r.text
    assert "Data editor" in body
    assert "Review queue" in body
    assert "/api/data-editor/projects/" in body


def test_preview_approve_reject_save_flow(client) -> None:
    pid = _project(client)
    path = _make_dataset([
        {"question": "Q1?", "answer": "A1", "source": "a.txt"},
        {
            "conversations": [
                {"from": "human", "value": "Hello"},
                {"from": "gpt", "value": "Hi there"},
            ],
            "filename": "b.md",
            "chunk": 2,
        },
    ])
    try:
        prev = client.get(
            f"/api/data-editor/projects/{pid}/preview",
            params={"dataset": path, "limit": 10, "offset": 0},
        )
        assert prev.status_code == 200, prev.text
        data = prev.json()
        assert data["total"] == 2
        assert len(data["rows"]) == 2

        row = client.get(
            f"/api/data-editor/projects/{pid}/row",
            params={"dataset": path, "index": 0},
        )
        assert row.status_code == 200
        assert row.json()["question"] == "Q1?"

        ok = client.post(
            f"/api/data-editor/projects/{pid}/approve",
            json={"dataset": path, "index": 0},
        )
        assert ok.status_code == 200
        assert ok.json()["ok"] is True

        rej = client.post(
            f"/api/data-editor/projects/{pid}/reject",
            json={"dataset": path, "index": 1},
        )
        assert rej.status_code == 200

        reviews = client.get(
            f"/api/data-editor/projects/{pid}/review",
            params={"dataset": path},
        )
        assert reviews.status_code == 200
        decisions = {r["row_index"]: r["decision"] for r in reviews.json()}
        assert decisions[0] == "approved"
        assert decisions[1] == "rejected"

        patched = client.patch(
            f"/api/data-editor/projects/{pid}/row",
            json={
                "dataset": path,
                "index": 0,
                "row": {"question": "Q1 edited?", "answer": "A1b", "source": "a.txt"},
            },
        )
        assert patched.status_code == 200
        assert patched.json()["ok"] is True

        again = client.get(
            f"/api/data-editor/projects/{pid}/row",
            params={"dataset": path, "index": 0},
        )
        assert again.json()["question"] == "Q1 edited?"

        batch = client.post(
            f"/api/data-editor/projects/{pid}/save",
            json={
                "dataset": path,
                "rows": [
                    {"question": "only", "answer": "one"},
                ],
            },
        )
        assert batch.status_code == 200
        assert batch.json()["total"] == 1
    finally:
        Path(path).unlink(missing_ok=True)


def test_preview_missing_file_404(client) -> None:
    pid = _project(client)
    r = client.get(
        f"/api/data-editor/projects/{pid}/preview",
        params={"dataset": "/tmp/does-not-exist-fts-editor.jsonl"},
    )
    assert r.status_code == 404
