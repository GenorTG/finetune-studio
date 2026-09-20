"""File workbench: built-in parsed-text editor + pipeline flags + RAG quick index.

Genor's rule 2026-09-20: manual file work must be possible for BOTH the
training-data and RAG flows — edit the parsed text in the browser, the edit
re-chunks the data-prep source, originals stay immutable and viewable.
"""
from __future__ import annotations

import secrets
from pathlib import Path

import pytest

from finetune_studio.data.fs import file_library as fl


@pytest.fixture
def fts_root(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    root = tmp_path / "fts"
    (root / "projects").mkdir(parents=True)
    monkeypatch.setattr("finetune_studio.data.fs.paths._ROOT", root)
    monkeypatch.setattr("finetune_studio.data.fs.paths._PROJECTS", root / "projects")
    fl._PARSED_CACHE.clear()
    return root


def _project(client) -> str:
    r = client.post("/api/projects", json={"name": "Workbench Test"})
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


def _doc(label: str) -> bytes:
    body = "\n\n".join(
        f"Section {i} of {label}: the archive number is TE-{i}71 and the "
        f"keeper swore on green wax in the year 18{40 + i} at Saltmere."
        for i in range(1, 9)
    )
    return f"# {label}\n\n{body}\n".encode()


def test_parsed_edit_roundtrip_and_override(fts_root: Path, client) -> None:
    pid = _project(client)
    before_md_raw = _doc("charter")
    fid = _upload(client, pid, "charter.md", before_md_raw)

    before = client.get(f"/api/projects/{pid}/files/{fid}/parsed").json()
    assert before["source"] in ("converted", "sibling")

    edited = before["parsed_md"] + "\n\nHAND-EDITED MARKER 42\n"
    r = client.put(f"/api/projects/{pid}/files/{fid}/parsed",
                   json={"text": edited})
    body = r.json()
    assert body["ok"] is True, body
    assert body["chars"] == len(edited)

    after = client.get(f"/api/projects/{pid}/files/{fid}/parsed").json()
    assert after["source"] == "override"
    assert "HAND-EDITED MARKER 42" in after["parsed_md"]
    # raw bytes must be untouched (the collision bug the first pass had)
    raw_versions = fl.list_versions(pid, fid)
    assert Path(raw_versions[0]["raw_path"]).read_bytes() == before_md_raw


def test_reparse_discards_override(fts_root: Path, client) -> None:
    pid = _project(client)
    fid = _upload(client, pid, "ledger.md", _doc("ledger"))
    original = client.get(f"/api/projects/{pid}/files/{fid}/parsed").json()["parsed_md"]
    client.put(f"/api/projects/{pid}/files/{fid}/parsed",
               json={"text": original + "\nOVERRIDE\n"})
    assert "OVERRIDE" in client.get(f"/api/projects/{pid}/files/{fid}/parsed").json()["parsed_md"]

    r = client.post(f"/api/projects/{pid}/files/{fid}/reparse")
    assert r.json()["override_removed"] is True
    back = client.get(f"/api/projects/{pid}/files/{fid}/parsed").json()
    assert "OVERRIDE" not in back["parsed_md"]


def test_edit_rechunks_data_prep_source(fts_root: Path, client) -> None:
    from finetune_studio.data import project_filesystem as pfs

    pid = _project(client)
    name = f"vane-{secrets.token_hex(4)}.md"
    fid = _upload(client, pid, name, _doc("vane"))
    pr = client.post(f"/api/projects/{pid}/data-prep/sources",
                     json={"file_id": fid})
    assert pr.status_code == 200, pr.text
    source_id = pr.json()["source"]["id"]

    edited = "Edited manual text. " + ("Saltmere tribunal clause. " * 200)
    r = client.put(f"/api/projects/{pid}/files/{fid}/parsed", json={"text": edited})
    body = r.json()
    assert body["rechunked"] is True
    assert body["chunk_count"] > 0, body

    src = pfs.read_qa_source(pid, source_id)
    assert src["parser"] == "manual-edit"
    assert src["status"] == "ready"
    assert src["chunk_count"] == body["chunk_count"]
    sha = src["sha256"]
    parsed_txt = pfs.file_dir(pid, sha) / "parsed.txt"
    assert "Edited manual text." in parsed_txt.read_text(encoding="utf-8")
    chunks = list((pfs.file_dir(pid, sha) / "chunks").glob("0*.txt"))
    assert len(chunks) == body["chunk_count"]


def test_pipeline_flags(fts_root: Path, client) -> None:
    pid = _project(client)
    # .csv is NOT auto-promoted on upload (.txt/.md/.markdown/.log are)
    fid_a = _upload(client, pid, f"a-{secrets.token_hex(3)}.csv", b"k,v\n1,2\n")
    fid_b = _upload(client, pid, f"b-{secrets.token_hex(3)}.md", _doc("beta"))
    client.post(f"/api/projects/{pid}/data-prep/sources", json={"file_id": fid_b})

    status = client.get(f"/api/projects/{pid}/files/pipeline").json()["status"]
    assert status[fid_a]["source_id"] is None
    assert status[fid_b]["source_id"], status[fid_b]
    assert status[fid_b]["has_parsed"] is True
    assert status[fid_b]["chunk_count"] > 0
    assert status[fid_b]["in_rag"] is False  # no corpus in sandbox


def test_rag_quick_promotes_then_builds(fts_root: Path, client, monkeypatch) -> None:
    from finetune_studio.webui.routes import rag as rag_mod

    calls: dict = {}

    async def fake_build(pid: str, req):
        calls["pid"] = pid
        calls["reset"] = req.reset
        return {"ok": True, "documents": 3, "chunks": 12}

    monkeypatch.setattr(rag_mod, "rag_build", fake_build)

    pid = _project(client)
    # .csv is not auto-promoted, so the quick flow has real work to do
    fid = _upload(client, pid, f"quick-{secrets.token_hex(3)}.csv",
                  b"question,answer\n" + _doc("quick"))
    r = client.post(f"/api/projects/{pid}/rag/quick", json={})
    body = r.json()
    assert body["ok"] is True, body
    assert body["promoted"] == 1
    assert body["promoted_files"][0]["file_id"] == fid
    assert body["build"]["chunks"] == 12
    assert calls["pid"] == pid

    # second run: already a source → nothing left to promote
    body2 = client.post(f"/api/projects/{pid}/rag/quick", json={}).json()
    assert body2["promoted"] == 0


def test_parsed_put_validation(fts_root: Path, client) -> None:
    pid = _project(client)
    fid = _upload(client, pid, f"v-{secrets.token_hex(3)}.md", _doc("val"))
    assert client.put(f"/api/projects/{pid}/files/{fid}/parsed",
                      json={}).status_code == 400
    assert client.put(f"/api/projects/nope/files/{fid}/parsed",
                      json={"text": "x"}).status_code == 404


def test_pipeline_in_rag_from_manifest(fts_root: Path, client) -> None:
    """Corpus document ids are md5(path) — the badge must match the sha12
    directory inside documents_meta[].source instead."""
    import json as _json

    pid = _project(client)
    fid = _upload(client, pid, f"rag-{secrets.token_hex(3)}.md", _doc("ragbadge"))
    status = client.get(f"/api/projects/{pid}/files/pipeline").json()["status"]
    sha12 = status[fid]["source_id"]  # auto-promoted on .md upload
    assert sha12
    assert status[fid]["in_rag"] is False

    corpus = fts_root / "rag_corpora" / pid
    corpus.mkdir(parents=True)
    (corpus / "manifest.json").write_text(_json.dumps({
        "extra": {"documents_meta": [
            {"document_id": "deadbeefcafe", "source": f"{fts_root}/projects/{pid}/files/{sha12}/parsed.txt"},
        ]},
    }), encoding="utf-8")

    status2 = client.get(f"/api/projects/{pid}/files/pipeline").json()["status"]
    assert status2[fid]["in_rag"] is True


def test_data_page_renders_workbench_controls(client) -> None:
    pid = _project(client)
    r = client.get(f"/projects/{pid}/data")
    assert r.status_code == 200
    html = r.text
    assert "wb-editor-modal" in html
    assert "wb-tab-parsed" in html
    assert "wb-bulk-prep" in html
    assert "const PID = " in html
