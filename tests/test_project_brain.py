"""Tests for RAG docs-indexed panel + inventory / chunks / rebuild API."""
from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import patch

import pandas as pd
import pytest

from tests.test_brain_server_package import (
    corpus_dir as _full_corpus,  # noqa: F401  (fixture)
)


def _project(client) -> str:
    r = client.post("/api/projects", json={"name": "RAG Docs Panel Test"})
    assert r.status_code in (200, 201), r.text
    return r.json()["id"]


def _write_fake_corpus(root: Path, *, doc_id: str = "abc123def456") -> Path:
    """Minimal BrainIndex-shaped corpus: manifest + sources + chunks.parquet."""
    root.mkdir(parents=True, exist_ok=True)
    (root / "sources").mkdir(exist_ok=True)
    text = "Hello world. " * 20
    (root / "sources" / f"{doc_id}.txt").write_text(text, encoding="utf-8")
    chunks = pd.DataFrame([
        {
            "id": f"{doc_id}_0",
            "document_id": doc_id,
            "chunk_index": 0,
            "source": f"/tmp/{doc_id}.txt",
            "filename": "manual_notes.txt",
            "text": text[:80],
        },
        {
            "id": f"{doc_id}_1",
            "document_id": doc_id,
            "chunk_index": 1,
            "source": f"/tmp/{doc_id}.txt",
            "filename": "manual_notes.txt",
            "text": text[40:120],
        },
    ])
    chunks.to_parquet(root / "chunks.parquet", index=False)
    manifest = {
        "name": "test",
        "version": "2",
        "created_at": 1_700_000_000.0,
        "updated_at": 1_700_000_100.0,
        "embedding_model": {"name": "x", "dim": 8},
        "chunk_settings": {"size": 400, "overlap": 80, "splitter": "word"},
        "rag_settings": {},
        "documents": 1,
        "chunks": 2,
        "extra": {},
    }
    (root / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    return root


def test_brain_page_renders_docs_indexed_panel(
    client, tmp_path: Path, monkeypatch
) -> None:
    pid = _project(client)
    corpus = tmp_path / "corpora" / pid
    _write_fake_corpus(corpus)

    monkeypatch.setattr(
        "finetune_studio.webui.routes.brain_documents.corpus_dir",
        lambda p: corpus if p == pid else tmp_path / "corpora" / p,
    )

    r = client.get(f"/projects/{pid}/brain")
    assert r.status_code == 200, r.text
    body = r.text
    assert 'id="brain-docs-panel"' in body
    assert 'id="brain-docs-table"' in body
    for col in ("Doc name", "Type", "Chunks", "Status", "Last indexed", "Actions"):
        assert f">{col}</th>" in body
    assert "manual_notes.txt" in body
    assert "Rebuild" in body
    assert "View chunks" in body
    assert "rebuildDoc" not in body
    assert "viewDocChunks" in body
    assert 'id="brain-chunks-modal"' in body
    assert "1 document" in body
    assert "2 chunk" in body


def test_brain_page_empty_state(client) -> None:
    pid = _project(client)
    with patch(
        "finetune_studio.webui.routes.brain_documents.list_indexed_docs",
        return_value=[],
    ):
        r = client.get(f"/projects/{pid}/brain")
    assert r.status_code == 200
    assert "No documents indexed yet" in r.text
    assert 'id="brain-docs-panel"' in r.text


def test_list_indexed_docs_helper(tmp_path: Path, monkeypatch) -> None:
    from finetune_studio.webui.routes import brain_documents as pr

    pid = "pidtest01"
    corpus = tmp_path / "c" / pid
    _write_fake_corpus(corpus, doc_id="docAAA111222")
    monkeypatch.setattr(pr, "corpus_dir", lambda p: corpus)

    docs = pr.list_indexed_docs(pid)
    assert len(docs) == 1
    d = docs[0]
    assert d["id"] == "docAAA111222"
    assert d["name"] == "manual_notes.txt"
    assert d["chunks"] == 2
    assert d["status"] == "indexed"
    assert d["mime"] == "text/plain"
    assert pr.total_chunk_count(docs) == 2


def test_docs_api_returns_inventory(client, tmp_path: Path, monkeypatch) -> None:
    pid = _project(client)
    corpus = tmp_path / "corpora" / pid
    _write_fake_corpus(corpus)
    monkeypatch.setattr(
        "finetune_studio.webui.routes.brain_documents.corpus_dir",
        lambda p: corpus if p == pid else tmp_path / "missing" / p,
    )
    r = client.get(f"/api/projects/{pid}/brain/documents")
    assert r.status_code == 200, r.text
    data = r.json()
    assert data["document_count"] == 1
    assert data["chunk_count"] == 2
    assert data["docs"][0]["name"] == "manual_notes.txt"
    assert data["docs"][0]["status"] == "indexed"


def test_chunks_api_returns_previews(client, tmp_path: Path, monkeypatch) -> None:
    pid = _project(client)
    corpus = tmp_path / "corpora" / pid
    doc_id = "chunkdoc999"
    _write_fake_corpus(corpus, doc_id=doc_id)
    monkeypatch.setattr(
        "finetune_studio.webui.routes.brain_documents.corpus_dir",
        lambda p: corpus if p == pid else tmp_path / "missing" / p,
    )
    r = client.get(f"/api/projects/{pid}/brain/documents/{doc_id}/chunks")
    assert r.status_code == 200, r.text
    data = r.json()
    assert data["count"] == 2
    assert data["chunks"][0]["chunk_id"] == f"{doc_id}_0"
    assert len(data["chunks"][0]["preview"]) <= 120
    assert data["chunks"][0]["preview"].startswith("Hello")


def test_build_then_documents_list_the_index(client, tmp_path: Path, monkeypatch) -> None:
    """One build route (the per-document "rebuild" that rebuilt everything is gone); the documents list follows it."""
    from tests.knowledge_helpers import add_source

    pid = _project(client)
    add_source(pid, "a.txt", "alpha beta gamma, one parsed sentence")

    class _FakeIndex:
        def __init__(self, d: Path) -> None:
            self.dir = Path(d)

        def build_from_documents(self, docs, **kwargs: object) -> dict:
            assert [d.filename for d in docs] == ["a.txt"]
            _write_fake_corpus(self.dir)
            return {"documents": 1, "chunks": 2, "skipped": 0}

    monkeypatch.setattr("finetune_studio.smart_brain.BrainIndex", _FakeIndex)
    r = client.post(f"/api/projects/{pid}/brain/build", json={"reset": True})
    from tests.knowledge_helpers import wait_build
    b = wait_build(client, pid, r)
    assert b["ok"] and b["documents"] == 1, b
    docs = client.get(f"/api/projects/{pid}/brain/documents").json()
    assert docs["document_count"] == 1 and docs["chunk_count"] == 2
    assert client.post(f"/api/projects/{pid}/brain/rebuild", json={}).status_code == 404


def test_docs_api_404_unknown_project(client) -> None:
    r = client.get("/api/projects/doesnotexist/brain/documents")
    assert r.status_code == 404


# ---------------------------------------------------------------- export API
@pytest.fixture()
def export_env(client, tmp_path: Path, monkeypatch, _full_corpus: Path):  # noqa: F811
    """Project wired to a full on-disk corpus, isolated settings + output dir."""
    from finetune_studio.webui.routes import settings as settings_mod

    pid = _project(client)
    monkeypatch.setattr(settings_mod, "SETTINGS_PATH", tmp_path / "settings.json")
    monkeypatch.setattr(
        "finetune_studio.webui.routes.brain_export._corpus_dir", lambda p: _full_corpus)
    monkeypatch.chdir(tmp_path)  # packages land in ./output/projects/<pid>/
    return pid


def test_export_config_defaults_and_overrides(client, export_env) -> None:
    pid = export_env
    d = client.get(f"/api/projects/{pid}/brain/export-config").json()
    assert d["effective"]["encrypt"] is True and d["effective"]["host"] == "127.0.0.1"
    assert d["project_override"] == {}

    r = client.put(f"/api/projects/{pid}/brain/export-config",
                   json={"scope": "studio", "values": {"port": 9000, "top_k": 7}})
    assert r.status_code == 200, r.text
    r = client.put(f"/api/projects/{pid}/brain/export-config",
                   json={"scope": "project", "values": {"port": 9100}})
    eff = r.json()["effective"]
    assert (eff["port"], eff["top_k"]) == (9100, 7)
    assert r.json()["defaults"]["port"] == 9000
    r = client.put(f"/api/projects/{pid}/brain/export-config",
                   json={"scope": "project", "clear": True})
    assert r.json()["effective"]["port"] == 9000
    bad = client.put(f"/api/projects/{pid}/brain/export-config",
                     json={"scope": "studio", "values": {"device": "tpu"}})
    assert bad.status_code == 422
    assert client.put(f"/api/projects/{pid}/brain/export-config",
                      json={"scope": "nope"}).status_code == 400
    assert client.get("/api/projects/nope/brain/export-config").status_code == 404


def test_mcp_package_encrypted_flow(client, export_env, tmp_path: Path) -> None:
    pid = export_env
    client.put(f"/api/projects/{pid}/brain/export-config",
               json={"scope": "studio", "values": {"kdf_log_n": 10}})
    r = client.post(f"/api/projects/{pid}/brain/standalone", json={})
    assert r.status_code == 200, r.text
    assert r.headers["cache-control"] == "no-store"
    j = r.json()
    assert j["encrypted"] is True and j["passphrase"] and len(j["passphrase"]) >= 20
    assert "PLAINTEXT" not in j["filename"]

    assert j["download_url"].startswith("/api/projects/")
    dl = client.get(j["download_url"])
    assert dl.status_code == 200
    assert j["passphrase"].encode() not in dl.content
    assert b"Vaelindrath" not in dl.content
    # the passphrase is not persisted anywhere under settings/output
    for f in tmp_path.rglob("*"):
        if f.is_file() and f.suffix in (".json", ".txt", ".md"):
            assert j["passphrase"] not in f.read_text(errors="ignore")

    # supplied passphrase is not echoed back
    r2 = client.post(f"/api/projects/{pid}/brain/standalone",
                     json={"passphrase": "my own long passphrase", "archive_format": "zip"})
    assert r2.status_code == 200, r2.text
    assert r2.json()["passphrase"] is None and r2.json()["filename"].endswith(".zip")


def test_mcp_package_plaintext_is_explicit_and_labelled(client, export_env) -> None:
    pid = export_env
    r = client.post(f"/api/projects/{pid}/brain/standalone", json={"encrypt": False})
    assert r.status_code == 200, r.text
    j = r.json()
    assert j["encrypted"] is False and j["passphrase"] is None
    assert "PLAINTEXT" in j["filename"]
    # passphrase + encrypt=false is a contradiction, not a silent downgrade
    bad = client.post(f"/api/projects/{pid}/brain/standalone",
                      json={"encrypt": False, "passphrase": "some long passphrase"})
    assert bad.status_code == 422
    short = client.post(f"/api/projects/{pid}/brain/standalone", json={"passphrase": "x"})
    assert short.status_code == 422


def test_mcp_package_download_validation_and_old_get_gone(client, export_env) -> None:
    pid = export_env
    for bad in ("../../etc/passwd", "a/b.zip", "x.exe", "..%2f..%2fsecret.zip"):
        r = client.get(f"/api/projects/{pid}/brain/standalone/download", params={"file": bad})
        assert r.status_code == 400, bad
    assert client.get(f"/api/projects/{pid}/brain/standalone/download",
                      params={"file": "missing.zip"}).status_code == 404
    # the old GET build endpoint (passphrase-in-URL era / unencrypted) is gone
    assert client.get(f"/api/projects/{pid}/brain/standalone").status_code == 405


def test_mcp_package_no_corpus_404(client, tmp_path: Path, monkeypatch) -> None:
    pid = _project(client)
    monkeypatch.setattr(
        "finetune_studio.webui.routes.brain_export._corpus_dir", lambda p: tmp_path / "none")
    monkeypatch.chdir(tmp_path)
    assert client.post(f"/api/projects/{pid}/brain/standalone", json={}).status_code == 404
