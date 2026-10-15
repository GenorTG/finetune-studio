"""Tests for file-library APIs: parsed MD, rename, per-file purge."""
from __future__ import annotations

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
    return root


def _project(client) -> str:
    r = client.post("/api/projects", json={"name": "File Library API Test"})
    assert r.status_code in (200, 201), r.text
    return r.json()["id"]


def _upload(
    client,
    pid: str,
    name: str,
    content: bytes,
    content_type: str = "text/plain",
) -> str:
    r = client.post(
        f"/api/projects/{pid}/files/upload",
        files=[("files", (name, content, content_type))],
    )
    assert r.status_code == 200, r.text
    data = r.json()
    assert data["counts"]["uploaded"] == 1, data
    return data["report"][0]["file_id"]


def _unique(label: str) -> bytes:
    """Unique payload so sha256-based file ids never collide across tests."""
    import secrets
    return f"{label}-{secrets.token_hex(8)}".encode()


# ── GET /parsed (the file's one parse, knowledge.sources) ─────────────────


def test_parsed_txt(client, fts_root: Path) -> None:
    pid = _project(client)
    body = b"hello world\n" + _unique("txt")
    fid = _upload(client, pid, "notes.txt", body)
    r = client.get(f"/api/projects/{pid}/files/{fid}/parsed")
    assert r.status_code == 200, r.text
    data = r.json()
    assert data["fid"] == fid and data["source_id"] and data["parser"]
    assert "hello world" in data["parsed_md"]


def test_parsed_json(client, fts_root: Path) -> None:
    pid = _project(client)
    payload = f'{{"a": 1, "id": "{_unique("json").decode()}"}}'
    fid = _upload(client, pid, "data.json", payload.encode(), "application/json")
    r = client.get(f"/api/projects/{pid}/files/{fid}/parsed")
    assert r.status_code == 200, r.text
    assert '"a": 1' in r.json()["parsed_md"]


def test_parsed_csv_is_a_table(client, fts_root: Path) -> None:
    pid = _project(client)
    csv_body = f"name,age\nAda,36\n{_unique('csv').decode()},1\n"
    fid = _upload(client, pid, "rows.csv", csv_body.encode(), "text/csv")
    r = client.get(f"/api/projects/{pid}/files/{fid}/parsed")
    assert r.status_code == 200, r.text
    assert "name | age" in r.json()["parsed_md"] and "Ada | 36" in r.json()["parsed_md"]


def test_parsed_unreadable_pdf_422_with_the_reason(client, fts_root: Path) -> None:
    pid = _project(client)
    fid = _upload(client, pid, "scan.pdf", b"%PDF-1.4 binary\x00" + _unique("pdf"), "application/pdf")
    r = client.get(f"/api/projects/{pid}/files/{fid}/parsed")
    assert r.status_code == 422, r.text
    assert "scan.pdf" in r.json()["detail"] and "no text" in r.json()["detail"]


def test_parsed_404(client, fts_root: Path) -> None:
    pid = _project(client)
    r = client.get(f"/api/projects/{pid}/files/doesnotexist/parsed")
    assert r.status_code == 404


def test_parsed_is_the_stored_parse_until_reparse(client, fts_root: Path) -> None:
    pid = _project(client)
    fid = _upload(client, pid, "cache.txt", b"v1-" + _unique("cache"))
    r1 = client.get(f"/api/projects/{pid}/files/{fid}/parsed")
    assert r1.status_code == 200
    versions = fl.list_versions(pid, fid)
    Path(versions[0]["raw_path"]).write_text("v2-CHANGED on disk", encoding="utf-8")
    r2 = client.get(f"/api/projects/{pid}/files/{fid}/parsed")
    assert "v1" in r2.json()["parsed_md"] and "v2-CHANGED" not in r2.json()["parsed_md"]
    assert client.post(f"/api/projects/{pid}/files/{fid}/reparse").status_code == 200
    assert "v2-CHANGED" in client.get(f"/api/projects/{pid}/files/{fid}/parsed").json()["parsed_md"]


# ── PATCH /rename ────────────────────────────────────────────────────────


def test_rename_ok(client, fts_root: Path) -> None:
    pid = _project(client)
    fid = _upload(client, pid, "old.txt", _unique("rename-ok"))
    old_path = Path(fl.list_versions(pid, fid)[0]["raw_path"])
    assert old_path.exists()
    r = client.patch(
        f"/api/projects/{pid}/files/{fid}/rename",
        json={"new_name": "foo.txt"},
    )
    assert r.status_code == 200, r.text
    data = r.json()
    assert data["ok"] is True
    assert data["file"]["original_name"] == "foo.txt"
    assert data["file"]["id"] == fid
    new_path = Path(fl.list_versions(pid, fid)[0]["raw_path"])
    assert new_path.exists()
    assert not old_path.exists() or old_path == new_path


def test_rename_accepts_name_alias(client, fts_root: Path) -> None:
    """data-prep UI historically sent {name: ...}; accept both keys."""
    pid = _project(client)
    fid = _upload(client, pid, "a.txt", _unique("rename-alias"))
    r = client.patch(
        f"/api/projects/{pid}/files/{fid}/rename",
        json={"name": "b.txt"},
    )
    assert r.status_code == 200, r.text
    assert r.json()["file"]["original_name"] == "b.txt"


def test_rename_conflict_409(client, fts_root: Path) -> None:
    pid = _project(client)
    _upload(client, pid, "taken.txt", _unique("taken"))
    fid = _upload(client, pid, "other.txt", _unique("other"))
    r = client.patch(
        f"/api/projects/{pid}/files/{fid}/rename",
        json={"new_name": "taken.txt"},
    )
    assert r.status_code == 409, r.text


def test_rename_path_sep_400(client, fts_root: Path) -> None:
    pid = _project(client)
    fid = _upload(client, pid, "safe.txt", _unique("pathsep"))
    r = client.patch(
        f"/api/projects/{pid}/files/{fid}/rename",
        json={"new_name": "evil/../x.txt"},
    )
    assert r.status_code == 400, r.text


def test_rename_404(client, fts_root: Path) -> None:
    pid = _project(client)
    r = client.patch(
        f"/api/projects/{pid}/files/missing/rename",
        json={"new_name": "x.txt"},
    )
    assert r.status_code == 404


# ── POST /purge ──────────────────────────────────────────────────────────


def test_purge_requires_trash(client, fts_root: Path) -> None:
    pid = _project(client)
    fid = _upload(client, pid, "live.txt", _unique("live"))
    r = client.post(f"/api/projects/{pid}/files/{fid}/purge")
    assert r.status_code == 409, r.text


def test_purge_trashed_file(client, fts_root: Path) -> None:
    pid = _project(client)
    fid = _upload(client, pid, "gone.txt", _unique("gone"))
    raw = Path(fl.list_versions(pid, fid)[0]["raw_path"])
    assert raw.exists()
    d = client.delete(f"/api/projects/{pid}/files/{fid}")
    assert d.status_code == 200, d.text
    r = client.post(f"/api/projects/{pid}/files/{fid}/purge")
    assert r.status_code == 200, r.text
    data = r.json()
    assert data["ok"] is True
    assert data["fid"] == fid
    # Gone from listing (including deleted)
    listed = client.get(
        f"/api/projects/{pid}/files?include_deleted=true"
    ).json()
    assert all(f["id"] != fid for f in listed["files"])
    assert fl.get_file(pid, fid, include_deleted=True) is None


def test_purge_404(client, fts_root: Path) -> None:
    pid = _project(client)
    r = client.post(f"/api/projects/{pid}/files/nope/purge")
    assert r.status_code == 404


def test_parse_state_lives_on_file_conversions_not_project_files(client, fts_root: Path) -> None:
    """project_files holds the file; its one parse is the file_conversions row (format 'parsed')."""
    from finetune_studio import db

    with db.cursor() as c:
        cols = {r["name"] for r in c.execute("PRAGMA table_info(project_files)")}
        conv = {r["name"] for r in c.execute("PRAGMA table_info(file_conversions)")}
    assert not {"parsed_md", "parsed_path", "stored_path", "status"} & cols
    assert {"status", "char_count", "chunk_count", "edited_at"} <= conv


def test_list_files_total_count_unfiltered(client, fts_root: Path) -> None:
    """total_count is project-wide even when folder_id / search filter the page."""
    pid = _project(client)
    for i in range(3):
        _upload(client, pid, f"a{i}.txt", _unique(f"a{i}"))
    listed = client.get(f"/api/projects/{pid}/files").json()
    assert listed["count"] == 3
    assert listed["total_count"] == 3

    folders = client.get(f"/api/projects/{pid}/folders").json()["folders"]
    auto = next((f for f in folders if f.get("kind") == "auto"), None)
    assert auto is not None
    filtered = client.get(
        f"/api/projects/{pid}/files", params={"folder_id": auto["id"]}
    ).json()
    assert filtered["total_count"] == 3
    assert filtered["count"] == len(filtered["files"])
    # Search that matches nothing must not zero the project total.
    none = client.get(
        f"/api/projects/{pid}/files", params={"search": "zzznomatchzzz"}
    ).json()
    assert none["count"] == 0
    assert none["total_count"] == 3
    assert none["files"] == []
