"""Project archive export -> import round-trip contract and import hardening."""
from __future__ import annotations

import io
import json
import tarfile

import pytest


@pytest.fixture
def roots(client, tmp_path, monkeypatch):
    from finetune_studio.webui.routes import projects as routes

    proj, rag = tmp_path / "projects", tmp_path / "rag_corpora"
    monkeypatch.setattr(routes, "_projects_root", lambda: proj)
    monkeypatch.setattr(routes, "_corpora_root", lambda: rag)
    return proj, rag


def _tar(members: list[tuple[str, bytes | None, bytes]]) -> bytes:
    """members: (name, payload, type); payload None => no data."""
    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w:gz") as tar:
        for name, payload, typ in members:
            info = tarfile.TarInfo(name)
            info.type = typ
            if typ == tarfile.REGTYPE:
                info.size = len(payload or b"")
                tar.addfile(info, io.BytesIO(payload or b""))
            else:
                info.linkname = (payload or b"").decode()
                tar.addfile(info)
    return buf.getvalue()


def _import(client, data: bytes):
    return client.post("/api/projects/import", files={"file": ("a.tar.gz", data)})


def _manifest(pid="oldpid", **meta) -> tuple[str, bytes, bytes]:
    body = {"project_id": pid, "version": "2.0", "project": meta}
    return ("manifest.json", json.dumps(body).encode(), tarfile.REGTYPE)


def test_roundtrip_preserves_metadata_and_files(client, roots):
    proj, rag = roots
    pid = client.post("/api/projects", json={
        "name": "Src", "description": "d", "base_model": "m/x", "system_prompt": "sp"}).json()["id"]
    (proj / pid / "sub").mkdir(parents=True)
    (proj / pid / "sub" / "a.txt").write_text("hello")
    (rag / pid).mkdir(parents=True)
    (rag / pid / "idx.bin").write_bytes(b"\x00\x01")

    r = client.get(f"/api/projects/{pid}/export")
    assert r.status_code == 200
    out = _import(client, r.content)
    assert out.status_code == 200, out.text
    item = out.json()["imported"][0]
    new = item["new_id"]
    assert out.json()["count"] == 1 and item["old_id"] == pid and new != pid
    got = client.get(f"/api/projects/{new}").json()
    assert (got["name"], got["base_model"], got["system_prompt"]) == ("Src", "m/x", "sp")
    assert (proj / new / "sub" / "a.txt").read_text() == "hello"
    assert (rag / new / "idx.bin").read_bytes() == b"\x00\x01"


def test_export_unknown_project_404(client, roots):
    assert client.get("/api/projects/nope/export").status_code == 404


@pytest.mark.parametrize("name", [
    "projects/oldpid/../../evil.txt", "/etc/evil", "projects/oldpid/../../../evil"])
def test_import_rejects_path_traversal(client, roots, tmp_path, name):
    data = _tar([_manifest(), ("projects/oldpid/ok.txt", b"1", tarfile.REGTYPE),
                 (name, b"x", tarfile.REGTYPE)])
    before = len(client.get("/api/projects").json())
    r = _import(client, data)
    assert r.status_code == 400
    assert not (tmp_path / "evil.txt").exists()
    assert len(client.get("/api/projects").json()) == before


def test_import_rejects_symlink_member(client, roots):
    data = _tar([_manifest(), ("projects/oldpid/l", b"/etc", tarfile.SYMTYPE)])
    assert _import(client, data).status_code == 400


def test_import_rejects_foreign_project_data(client, roots):
    data = _tar([_manifest(), ("projects/oldpid/a", b"1", tarfile.REGTYPE),
                 ("projects/other/b", b"2", tarfile.REGTYPE)])
    assert _import(client, data).status_code == 400


def test_import_rejects_unexpected_top_level(client, roots):
    data = _tar([_manifest(), ("projects/oldpid/a", b"1", tarfile.REGTYPE),
                 ("etc/passwd", b"x", tarfile.REGTYPE)])
    assert _import(client, data).status_code == 400


def test_import_requires_projects_dir(client, roots):
    assert _import(client, _tar([_manifest()])).status_code == 400


def test_import_legacy_archive_without_metadata(client, roots):
    proj, _ = roots
    data = _tar([("projects/legacy1234/f.txt", b"z", tarfile.REGTYPE)])
    r = _import(client, data)
    assert r.status_code == 200, r.text
    new = r.json()["imported"][0]["new_id"]
    assert (proj / new / "f.txt").read_text() == "z"


def test_roundtrip_restores_file_library_and_rebases_source_paths(client, roots, tmp_path):
    """Imported project lists its files and its sources pass the path fence."""
    from finetune_studio import db
    proj, _rag = roots
    pid = client.post("/api/projects", json={"name": "Lib"}).json()["id"]
    old_raw = proj / pid / "files" / "raw" / "other" / f"{pid}-abc_a.txt"
    old_raw.parent.mkdir(parents=True)
    old_raw.write_text("hello")
    fid = f"{pid}-abc"
    with db.cursor() as c:
        c.execute("INSERT INTO file_folders (id, project_id, name, kind, parent_id, created_at)"
                  " VALUES ('fld_auto_other_1', ?, 'other', 'auto', NULL, 1)", (pid,))
        c.execute("INSERT INTO project_files (id, project_id, original_name, current_version,"
                  " size_bytes, uploaded_at) VALUES (?, ?, 'a.txt', 1, 5, 1)", (fid, pid))
        c.execute("INSERT INTO file_versions (file_id, version, raw_path, raw_hash, raw_size,"
                  " uploaded_at) VALUES (?, 1, ?, 'h', 5, 1)", (fid, str(old_raw)))
        c.execute("INSERT INTO folder_membership VALUES ('fld_auto_other_1', ?)", (fid,))
    src_dir = proj / pid / "qa" / "sources"
    src_dir.mkdir(parents=True)
    (src_dir / "s1.json").write_text(json.dumps(
        {"id": "s1", "filename": "a.txt", "data_path": str(old_raw)}))

    new = _import(client, client.get(f"/api/projects/{pid}/export").content).json()["imported"][0]["new_id"]

    listed = client.get(f"/api/projects/{new}/files").json()["files"]
    assert [f["original_name"] for f in listed] == ["a.txt"]
    assert listed[0]["id"] == f"{new}-abc"
    data_path = json.loads((proj / new / "qa" / "sources" / "s1.json").read_text())["data_path"]
    assert f"/projects/{new}/" in data_path and f"/projects/{pid}/" not in data_path
    assert (proj / new / "files" / "raw" / "other" / f"{pid}-abc_a.txt").is_file()
    assert len(client.get(f"/api/projects/{pid}/files").json()["files"]) == 1
