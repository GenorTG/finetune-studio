"""API + page smoke tests for the project-scoped data editor."""
from __future__ import annotations

import json
from pathlib import Path

import pytest
from fastapi import HTTPException

from finetune_studio.config import settings
from finetune_studio.db.datasets import datasets_dir
from finetune_studio.webui.routes import data_editor as de


def _write_dataset(pid: str, name: str, rows: list[dict]) -> Path:
    """Write a JSONL under the project's datasets dir; return absolute Path."""
    target = datasets_dir(pid) / name
    target.write_text(
        "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows),
        encoding="utf-8",
    )
    return target


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
    path = _write_dataset(pid, "suite.jsonl", [
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
    dataset = str(path)
    prev = client.get(
        f"/api/data-editor/projects/{pid}/preview",
        params={"dataset": dataset, "limit": 10, "offset": 0},
    )
    assert prev.status_code == 200, prev.text
    data = prev.json()
    assert data["total"] == 2
    assert len(data["rows"]) == 2

    row = client.get(
        f"/api/data-editor/projects/{pid}/row",
        params={"dataset": dataset, "index": 0},
    )
    assert row.status_code == 200
    assert row.json()["question"] == "Q1?"

    ok = client.post(
        f"/api/data-editor/projects/{pid}/approve",
        json={"dataset": dataset, "index": 0},
    )
    assert ok.status_code == 200
    assert ok.json()["ok"] is True

    rej = client.post(
        f"/api/data-editor/projects/{pid}/reject",
        json={"dataset": dataset, "index": 1},
    )
    assert rej.status_code == 200

    reviews = client.get(
        f"/api/data-editor/projects/{pid}/review",
        params={"dataset": dataset},
    )
    assert reviews.status_code == 200
    decisions = {r["row_index"]: r["decision"] for r in reviews.json()}
    assert decisions[0] == "approved"
    assert decisions[1] == "rejected"

    patched = client.patch(
        f"/api/data-editor/projects/{pid}/row",
        json={
            "dataset": dataset,
            "index": 0,
            "row": {"question": "Q1 edited?", "answer": "A1b", "source": "a.txt"},
        },
    )
    assert patched.status_code == 200
    assert patched.json()["ok"] is True

    again = client.get(
        f"/api/data-editor/projects/{pid}/row",
        params={"dataset": dataset, "index": 0},
    )
    assert again.json()["question"] == "Q1 edited?"

    batch = client.post(
        f"/api/data-editor/projects/{pid}/save",
        json={
            "dataset": dataset,
            "rows": [
                {"question": "only", "answer": "one"},
            ],
        },
    )
    assert batch.status_code == 200
    assert batch.json()["total"] == 1


def test_preview_missing_file_404(client) -> None:
    pid = _project(client)
    missing = datasets_dir(pid) / "does-not-exist-fts-editor.jsonl"
    r = client.get(
        f"/api/data-editor/projects/{pid}/preview",
        params={"dataset": str(missing)},
    )
    assert r.status_code == 404


def test_resolve_relative_under_data_dir(mock_settings, tmp_path, monkeypatch) -> None:
    """Normal relative form under data_dir resolves without double-prefix."""
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    monkeypatch.setattr(settings, "data_dir", str(data_dir))
    monkeypatch.setattr(settings, "db_path", str(data_dir / "finetune_studio.db"))
    from finetune_studio.db.connection import init_db
    init_db()

    pid = "rel-proj"
    target = datasets_dir(pid) / "rel.jsonl"
    target.write_text('{"q":1}\n', encoding="utf-8")
    rel = f"projects/{pid}/datasets/rel.jsonl"
    resolved = de._resolve_path(rel, pid=pid)
    assert resolved.exists()
    assert resolved.resolve() == target.resolve()
    assert "data/data/" not in resolved.as_posix()


def test_resolve_stored_data_projects_path(mock_settings, tmp_path, monkeypatch) -> None:
    """Stored ``data/projects/...`` must not become ``data/data/projects/...``."""
    cwd = tmp_path / "repo"
    cwd.mkdir()
    (cwd / "data").mkdir()
    monkeypatch.chdir(cwd)
    monkeypatch.setattr(settings, "data_dir", "data")
    monkeypatch.setattr(settings, "db_path", "data/finetune_studio.db")
    from finetune_studio.db.connection import init_db
    init_db()

    pid = "stored-proj"
    target = datasets_dir(pid) / "export.jsonl"
    target.write_text('{"q":1}\n', encoding="utf-8")
    stored = f"data/projects/{pid}/datasets/export.jsonl"
    assert Path(stored).exists()

    # Old join would point here and miss the file.
    buggy = Path(settings.data_dir) / stored
    assert buggy != Path(stored)
    assert not buggy.exists()

    resolved = de._resolve_path(stored, pid=pid)
    assert resolved.exists()
    assert resolved.resolve() == Path(stored).resolve()
    assert "data/data/" not in resolved.as_posix()


def test_resolve_absolute_under_project(client) -> None:
    pid = _project(client)
    path = _write_dataset(pid, "abs.jsonl", [{"ok": True}])
    resolved = de._resolve_path(str(path), pid=pid)
    assert resolved.resolve() == path.resolve()
    prev = client.get(
        f"/api/data-editor/projects/{pid}/preview",
        params={"dataset": str(path)},
    )
    assert prev.status_code == 200
    assert prev.json()["total"] == 1


def test_resolve_rejects_traversal(tmp_path, monkeypatch) -> None:
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    monkeypatch.setattr(settings, "data_dir", str(data_dir))
    monkeypatch.setattr(settings, "db_path", str(data_dir / "db.sqlite"))
    pid = "trav-proj"
    datasets_dir(pid)

    with pytest.raises(HTTPException) as excinfo:
        de._resolve_path(f"projects/{pid}/datasets/../../other/x.jsonl", pid=pid)
    assert excinfo.value.status_code == 400

    with pytest.raises(HTTPException) as excinfo2:
        de._resolve_path("../etc/passwd", pid=pid)
    assert excinfo2.value.status_code == 400


def test_resolve_rejects_cross_project(client) -> None:
    pid_a = _project(client)
    pid_b = _project(client)
    path_b = _write_dataset(pid_b, "secret.jsonl", [{"secret": True}])

    with pytest.raises(HTTPException) as excinfo:
        de._resolve_path(str(path_b), pid=pid_a)
    assert excinfo.value.status_code == 403

    forbidden = client.get(
        f"/api/data-editor/projects/{pid_a}/preview",
        params={"dataset": str(path_b)},
    )
    assert forbidden.status_code == 403
