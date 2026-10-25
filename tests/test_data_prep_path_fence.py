"""Data prep never reads or writes outside the project directory (Genor 2026-10-02).

Covers the shared helper (`data.fs.paths.resolve_within` / `resolve_in_project`)
and every data-prep endpoint that accepts a path reference.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from fastapi import HTTPException
from fastapi.testclient import TestClient

from finetune_studio import db
from finetune_studio.config import settings
from finetune_studio.data.fs import paths
from finetune_studio.webui.app import app


@pytest.fixture
def env(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    """(client, project_id, project_dir, outside_dir) with FTS root + DB in tmp."""
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    monkeypatch.setattr(settings, "db_path", str(data_dir / "fence.db"))
    monkeypatch.setattr(settings, "data_dir", str(data_dir))
    root = tmp_path / "fts_root"
    (root / "projects").mkdir(parents=True)
    monkeypatch.setattr(paths, "_ROOT", root)
    monkeypatch.setattr(paths, "_PROJECTS", root / "projects")
    db.init_db()
    client = TestClient(app)
    r = client.post("/api/projects", json={"name": "fence", "base_model": "x/test"})
    assert r.status_code == 200, r.text
    pid = r.json()["id"]
    outside = tmp_path / "outside"
    outside.mkdir()
    return client, pid, paths.project_dir(pid), outside


def _jsonl(path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"question": "q", "answer": "a"}) + "\n", encoding="utf-8")
    return path


# ── helper contract ──────────────────────────────────────────────────────────

def test_helper_rejects_absolute_outside(env) -> None:
    _, pid, _, outside = env
    f = _jsonl(outside / "x.jsonl")
    with pytest.raises(HTTPException) as e:
        paths.resolve_in_project(pid, str(f))
    assert e.value.status_code == 403


def test_helper_rejects_dotdot(env) -> None:
    _, pid, pdir, _ = env
    with pytest.raises(HTTPException) as e:
        paths.resolve_in_project(pid, str(pdir / ".." / "x.jsonl"))
    assert e.value.status_code == 400
    with pytest.raises(HTTPException) as e2:
        paths.resolve_in_project(pid, "../x.jsonl")
    assert e2.value.status_code == 400


def test_helper_rejects_symlink_escape(env) -> None:
    _, pid, pdir, outside = env
    target = _jsonl(outside / "secret.jsonl")
    link = pdir / "link.jsonl"
    link.symlink_to(target)
    with pytest.raises(HTTPException) as e:
        paths.resolve_in_project(pid, str(link))
    assert e.value.status_code == 403
    (pdir / "dirlink").symlink_to(outside, target_is_directory=True)
    with pytest.raises(HTTPException) as e2:
        paths.resolve_in_project(pid, "dirlink/secret.jsonl")
    assert e2.value.status_code == 403


def test_helper_accepts_inside_absolute_and_relative(env) -> None:
    _, pid, pdir, _ = env
    f = _jsonl(pdir / "files" / "ok.jsonl")
    assert paths.resolve_in_project(pid, str(f)) == f.resolve()
    assert paths.resolve_in_project(pid, "files/ok.jsonl") == f.resolve()


def test_helper_rejects_empty_and_other_project(env) -> None:
    client, pid, _, _ = env
    with pytest.raises(HTTPException) as e:
        paths.resolve_in_project(pid, "  ")
    assert e.value.status_code == 400
    other = client.post("/api/projects", json={"name": "other", "base_model": "x/test"}).json()["id"]
    f = _jsonl(paths.project_dir(other) / "theirs.jsonl")
    with pytest.raises(HTTPException) as e2:
        paths.resolve_in_project(pid, str(f))
    assert e2.value.status_code == 403


# ── endpoints ────────────────────────────────────────────────────────────────

def test_data_editor_rejects_outside_and_accepts_inside(env) -> None:
    client, pid, pdir, outside = env
    ext = _jsonl(outside / "ext.jsonl")
    url = f"/api/data-editor/projects/{pid}/preview"
    assert client.get(url, params={"dataset": str(ext)}).status_code == 403
    assert client.get(url, params={"dataset": "../ext.jsonl"}).status_code == 400
    link = pdir / "datasets" / "link.jsonl"
    link.parent.mkdir(parents=True, exist_ok=True)
    link.symlink_to(ext)
    assert client.get(url, params={"dataset": str(link)}).status_code == 403
    inside = _jsonl(pdir / "datasets" / "in.jsonl")
    ok = client.get(url, params={"dataset": str(inside)})
    assert ok.status_code == 200, ok.text
    # writes are fenced too: an edit aimed outside must not create/modify the file
    before = ext.read_text()
    r = client.patch(
        f"/api/data-editor/projects/{pid}/row",
        json={"dataset": str(ext), "index": 0, "row": {"question": "x", "answer": "y"}},
    )
    assert r.status_code == 403, r.text
    r = client.post(
        f"/api/data-editor/projects/{pid}/save",
        json={"dataset": str(outside / "new.jsonl"), "rows": [{"question": "x"}]},
    )
    assert r.status_code == 403, r.text
    assert not (outside / "new.jsonl").exists()
    assert ext.read_text() == before


def test_datasets_register_rejects_outside(env) -> None:
    client, pid, pdir, outside = env
    ext = _jsonl(outside / "ext.jsonl")
    url = f"/api/projects/{pid}/datasets"
    r = client.post(url, json={"data_path": str(ext)})
    assert r.status_code == 403, r.text
    assert client.post(url, json={"data_path": "../ext.jsonl"}).status_code == 400
    link = pdir / "datasets" / "l.jsonl"
    link.parent.mkdir(parents=True, exist_ok=True)
    link.symlink_to(ext)
    assert client.post(url, json={"data_path": str(link)}).status_code == 403
    inside = _jsonl(pdir / "datasets" / "in.jsonl")
    ok = client.post(url, json={"data_path": str(inside)})
    assert ok.status_code == 200, ok.text
    assert db.list_datasets(pid)[0]["data_path"] == str(inside.resolve())


def test_data_prep_sources_never_take_a_path(env) -> None:
    """Files enter only through the file library (2026-10-14): any ``data_path`` body is a 400, inside or out."""
    client, pid, pdir, outside = env
    ext = outside / "ext.txt"
    ext.write_text("external text that must not be ingested " * 5, encoding="utf-8")
    inside = pdir / "in.txt"
    inside.write_text("inside text is not taken by path either " * 5, encoding="utf-8")
    url = f"/api/projects/{pid}/data-prep/sources"
    for ref in (str(ext), "../ext.txt", str(inside)):
        assert client.post(url, json={"data_path": ref}).status_code == 400
    assert client.get(url).json()["sources"] == []


def test_file_library_raw_download_fenced(env) -> None:
    client, pid, _, outside = env
    up = client.post(
        f"/api/projects/{pid}/files/upload",
        files={"files": ("a.txt", b"hello", "text/plain")},
    )
    assert up.status_code == 200, up.text
    fid = up.json()["report"][0]["file_id"]
    assert client.get(f"/api/projects/{pid}/files/{fid}/raw").status_code == 200
    ext = outside / "stolen.txt"
    ext.write_text("secret", encoding="utf-8")
    with db.cursor() as c:
        c.execute("UPDATE file_versions SET raw_path = ? WHERE file_id = ?", (str(ext), fid))
    r = client.get(f"/api/projects/{pid}/files/{fid}/raw")
    assert r.status_code == 403


