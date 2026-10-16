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


def test_parsed_edit_roundtrip_keeps_raw_bytes(fts_root: Path, client) -> None:
    pid = _project(client)
    before_md_raw = _doc("charter")
    fid = _upload(client, pid, "charter.md", before_md_raw)

    before = client.get(f"/api/projects/{pid}/files/{fid}/parsed").json()
    assert before["parser"] and before["edited_at"] is None

    edited = before["parsed_md"] + "\n\nHAND-EDITED MARKER 42\n"
    r = client.put(f"/api/projects/{pid}/files/{fid}/parsed",
                   json={"text": edited})
    body = r.json()
    assert body["ok"] is True, body
    assert body["chars"] == len(edited)

    after = client.get(f"/api/projects/{pid}/files/{fid}/parsed").json()
    assert after["parser"] == "manual-edit" and after["edited_at"]
    assert "HAND-EDITED MARKER 42" in after["parsed_md"]
    # raw bytes must be untouched (the collision bug the first pass had)
    raw_versions = fl.list_versions(pid, fid)
    assert Path(raw_versions[0]["raw_path"]).read_bytes() == before_md_raw


def test_reparse_discards_edit(fts_root: Path, client) -> None:
    pid = _project(client)
    fid = _upload(client, pid, "ledger.md", _doc("ledger"))
    original = client.get(f"/api/projects/{pid}/files/{fid}/parsed").json()["parsed_md"]
    client.put(f"/api/projects/{pid}/files/{fid}/parsed",
               json={"text": original + "\nOVERRIDE\n"})
    assert "OVERRIDE" in client.get(f"/api/projects/{pid}/files/{fid}/parsed").json()["parsed_md"]

    r = client.post(f"/api/projects/{pid}/files/{fid}/reparse")
    assert r.status_code == 200, r.text
    back = client.get(f"/api/projects/{pid}/files/{fid}/parsed").json()
    assert "OVERRIDE" not in back["parsed_md"] and back["edited_at"] is None


def test_edit_rechunks_the_source(fts_root: Path, client) -> None:
    from finetune_studio.knowledge import sources

    pid = _project(client)
    name = f"vane-{secrets.token_hex(4)}.md"
    fid = _upload(client, pid, name, _doc("vane"))

    edited = "Edited manual text. " + ("Saltmere tribunal clause. " * 200)
    r = client.put(f"/api/projects/{pid}/files/{fid}/parsed", json={"text": edited})
    body = r.json()
    assert body["chunk_count"] > 0, body

    src = sources.source_for_file(pid, fid)
    assert src is not None and src.id == body["source_id"]
    assert src.parser == "manual-edit"
    assert src.status == "ready"
    assert src.chunk_count == body["chunk_count"]
    assert "Edited manual text." in sources.parsed_text(pid, src)
    assert len(sources.passages(pid, src.sha256)) == body["chunk_count"]


def test_parsed_view_reports_unreadable_and_unparsed_files(fts_root: Path, client) -> None:
    pid = _project(client)
    fid = _upload(client, pid, "blob.bin", b"\x00\x01binary")
    r = client.get(f"/api/projects/{pid}/files/{fid}/parsed")
    assert r.status_code == 422 and "no parser" in r.json()["detail"]


def test_pipeline_flags(fts_root: Path, client) -> None:
    pid = _project(client)
    # Every parser-supported upload is staged and parsed in the background.
    fid_a = _upload(client, pid, f"a-{secrets.token_hex(3)}.csv", b"k,v\n1,2\n")
    fid_b = _upload(client, pid, f"b-{secrets.token_hex(3)}.md", _doc("beta"))

    status = client.get(f"/api/projects/{pid}/files/pipeline").json()["status"]
    assert status[fid_a]["source_id"], status[fid_a]
    assert status[fid_a]["has_parsed"] is True
    assert status[fid_b]["source_id"], status[fid_b]
    assert status[fid_b]["has_parsed"] is True
    assert status[fid_b]["chunk_count"] > 0
    assert status[fid_b]["in_brain"] is False  # no corpus in sandbox


def test_brain_quick_promotes_then_builds(fts_root: Path, client, monkeypatch) -> None:
    """The quick index is one build job in mode ``quick``: the parse step runs in the job (2026-10-16), the
    finished build reports ``promoted`` / ``failed``."""
    from tests.knowledge_helpers import wait_build

    class _Index:
        def __init__(self, d: Path) -> None:
            self.dir = Path(d)

        def build_from_documents(self, docs, **_kw: object) -> dict:
            return {"documents": len(docs), "chunks": 12, "vector_dim": 8, "skipped": 0}

    monkeypatch.setattr("finetune_studio.smart_brain.BrainIndex", _Index)

    pid = _project(client)
    # Upload already staged and parsed the CSV; quick RAG can build directly.
    fid = _upload(client, pid, f"quick-{secrets.token_hex(3)}.csv",
                  b"question,answer\n" + _doc("quick"))
    r = client.post(f"/api/projects/{pid}/brain/quick", json={})
    assert r.status_code == 202, r.text
    body = r.json()
    assert body["ok"] is True and body["pending_parses"] == 0 and body["build"]["mode"] == "quick", body
    done = wait_build(client, pid, r)
    assert done["ok"] and done["mode"] == "quick" and done["chunks"] == 12
    assert done["promoted"] == 0 and done["promoted_files"] == [] and done["failed"] == []
    assert client.get(f"/api/projects/{pid}/files/{fid}/usage").json()["source"]

    # second run: already a source → nothing left to promote
    r2 = client.post(f"/api/projects/{pid}/brain/quick", json={})
    assert r2.json()["pending_parses"] == 0 and wait_build(client, pid, r2)["promoted"] == 0


def test_parsed_put_validation(fts_root: Path, client) -> None:
    pid = _project(client)
    fid = _upload(client, pid, f"v-{secrets.token_hex(3)}.md", _doc("val"))
    assert client.put(f"/api/projects/{pid}/files/{fid}/parsed",
                      json={}).status_code == 400
    assert client.put(f"/api/projects/nope/files/{fid}/parsed",
                      json={"text": "x"}).status_code == 404


def test_pipeline_in_brain_and_stale_from_the_index_manifest(fts_root: Path, client) -> None:
    """The badge reads ``built_from`` (source id -> text hash): a later edit marks the file stale, not missing."""
    import json as _json

    from finetune_studio.knowledge import sources

    pid = _project(client)
    fid = _upload(client, pid, f"rag-{secrets.token_hex(3)}.md", _doc("ragbadge"))
    status = client.get(f"/api/projects/{pid}/files/pipeline").json()["status"]
    sid = status[fid]["source_id"]
    assert sid and status[fid]["in_brain"] is False

    corpus = fts_root / "brain" / pid
    corpus.mkdir(parents=True)
    text_sha = sources.get_source(pid, sid).text_sha
    (corpus / "manifest.json").write_text(_json.dumps({"extra": {"built_from": {sid: text_sha}}}), encoding="utf-8")
    status2 = client.get(f"/api/projects/{pid}/files/pipeline").json()["status"]
    assert status2[fid]["in_brain"] is True and status2[fid]["brain_stale"] is False

    client.put(f"/api/projects/{pid}/files/{fid}/parsed", json={"text": "An edited text after the build."})
    status3 = client.get(f"/api/projects/{pid}/files/pipeline").json()["status"]
    assert status3[fid]["in_brain"] is True and status3[fid]["brain_stale"] is True


def test_data_page_renders_workbench_controls(client) -> None:
    pid = _project(client)
    r = client.get(f"/projects/{pid}/data")
    assert r.status_code == 200
    html = r.text
    assert "wb-editor-modal" in html
    assert "wb-tab-parsed" in html
    assert "wb-bulk-prep" in html
    assert "const PID = " in html
