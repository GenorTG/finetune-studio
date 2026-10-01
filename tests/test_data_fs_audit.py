"""Regression tests for discrepancies found in the data/fs documentation audit.

Covers:
- purge_trash() actually removing bytes from .RAW_TRASH / .CONVERTED_TRASH
  on disk, not just the DB rows (the bytes used to leak forever because
  soft_delete_file() never updates file_versions.raw_path /
  file_conversions.converted_path to point at the post-move trash location,
  so purge_trash()'s existence check against the stale DB path was always
  False and the unlink() branch was dead code).
"""
from __future__ import annotations

import time
from pathlib import Path

import pytest

from finetune_studio.data.fs import file_library as fl


@pytest.fixture
def fts_root(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Point the file-library disk root at a temp dir for isolated tests."""
    root = tmp_path / "fts"
    root.mkdir()
    projects = root / "projects"
    projects.mkdir()
    monkeypatch.setattr("finetune_studio.data.fs.paths._ROOT", root)
    monkeypatch.setattr("finetune_studio.data.fs.paths._PROJECTS", projects)
    fl._PARSED_CACHE.clear()
    return root


def _project(client) -> str:
    r = client.post("/api/projects", json={"name": "Data FS Audit Test"})
    assert r.status_code in (200, 201), r.text
    return r.json()["id"]


def _upload(client, pid: str, name: str, content: bytes) -> str:
    r = client.post(
        f"/api/projects/{pid}/files/upload",
        files=[("files", (name, content, "text/plain"))],
    )
    assert r.status_code == 200, r.text
    data = r.json()
    assert data["counts"]["uploaded"] == 1, data
    return data["report"][0]["file_id"]


def test_purge_trash_removes_bytes_from_raw_trash(client, fts_root: Path) -> None:
    """purge_trash() must delete the physical bytes it claims to purge.

    Before the fix: soft_delete_file() moves the raw file into
    files/raw/.RAW_TRASH/ but never updates file_versions.raw_path, so
    purge_trash() checked existence of the OLD (pre-trash) path — which
    never exists post-delete — and its unlink() never ran. The DB row was
    removed (reporting "purged": [...]) while the bytes leaked on disk
    forever.
    """
    import secrets

    pid = _project(client)
    content = f"purge-trash-{secrets.token_hex(8)}".encode()
    fid = _upload(client, pid, "to-be-purged.txt", content)

    # raw_path in DB still points at the pre-trash location by design.
    pre_trash_path = Path(fl.list_versions(pid, fid)[0]["raw_path"])
    assert pre_trash_path.exists()

    fl.soft_delete_file(pid, fid)

    # After soft-delete the DB-recorded path is stale (file physically moved).
    assert not pre_trash_path.exists()
    trash_dir = fl.raw_trash_dir(pid)
    trashed = list(trash_dir.glob(f"{fid}_*"))
    assert len(trashed) == 1, "soft_delete_file should have moved the file into .RAW_TRASH"
    trashed_path = trashed[0]
    assert trashed_path.exists()

    # Backdate deleted_at so it's eligible for purge_trash's age cutoff.
    from finetune_studio import db
    with db.cursor() as c:
        c.execute(
            "UPDATE project_files SET deleted_at = ? WHERE id = ?",
            (time.time() - 8 * 86400, fid),
        )

    result = fl.purge_trash(pid, older_than_days=7)

    assert result["count"] == 1
    assert not trashed_path.exists(), (
        "purge_trash() reported success but left the trashed bytes on disk"
    )
    assert result.get("disk_errors") == []
    assert fl.get_file(pid, fid, include_deleted=True) is None
