"""Project archive export -> import round-trip contract and import hardening."""
from __future__ import annotations

import io
import json
import tarfile
from pathlib import Path

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


def test_roundtrip_restores_files_with_their_parse(client, roots, monkeypatch):
    """An imported project lists its files, and each file's one parse comes along (knowledge.sources)."""
    from finetune_studio.data.fs import paths
    from finetune_studio.knowledge import sources
    from tests.knowledge_helpers import add_source

    proj, _rag = roots
    monkeypatch.setattr(paths, "_PROJECTS", proj)
    pid = client.post("/api/projects", json={"name": "Lib"}).json()["id"]
    src = add_source(pid, "a.txt", "Hello from the archive: the gate closes at 22:00.")
    sources.set_text(pid, src.id, "Hand-edited: the gate closes at 22:30.")

    new = _import(client, client.get(f"/api/projects/{pid}/export").content).json()["imported"][0]["new_id"]

    listed = client.get(f"/api/projects/{new}/files").json()["files"]
    assert [f["original_name"] for f in listed] == ["a.txt"]
    moved = sources.get_source(new, src.id)
    assert moved is not None and moved.ready and moved.parser == "manual-edit"
    assert f"/projects/{new}/" in moved.raw_path and Path(moved.raw_path).is_file()
    assert "22:30" in sources.parsed_text(new, moved)
    assert len(client.get(f"/api/projects/{pid}/files").json()["files"]) == 1


def test_import_moves_an_old_archive_off_qa_sources(client, roots, monkeypatch):
    """An archive from before 2026-10-14 carries qa/sources/*.json and no parse rows: the import migrates it."""
    from finetune_studio import db
    from finetune_studio.data.fs import paths
    from finetune_studio.data.fs.chunks import write_chunks
    from finetune_studio.knowledge import sources
    from tests.knowledge_helpers import add_source

    proj, _rag = roots
    monkeypatch.setattr(paths, "_PROJECTS", proj)
    pid = client.post("/api/projects", json={"name": "Old"}).json()["id"]
    src = add_source(pid, "a.txt", "Old layout text that was parsed long ago.", parse=False)
    d = sources.source_dir(pid, src)
    (d / "parsed.txt").write_text("Old layout text that was parsed long ago.", encoding="utf-8")
    write_chunks(pid, src.sha256, ["Old layout text that was parsed long ago."])
    (d / "metadata.json").write_text("{}", encoding="utf-8")
    db.file_parses.delete(src.file_id)                                          # no parse row, like an old archive
    (proj / pid / "qa" / "sources").mkdir(parents=True)
    (proj / pid / "qa" / "sources" / f"{src.id}.json").write_text(json.dumps(
        {"id": src.id, "file_id": src.file_id, "filename": "a.txt", "status": "ready", "parser": "text_v1"}))

    new = _import(client, client.get(f"/api/projects/{pid}/export").content).json()["imported"][0]["new_id"]

    moved = sources.get_source(new, src.id)
    assert moved is not None and moved.ready and moved.parser == "text_v1" and moved.chunk_count == 1
    assert not (proj / new / "qa" / "sources").exists()
    assert not (sources.source_dir(new, moved) / "metadata.json").exists()
