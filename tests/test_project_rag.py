"""Tests for RAG docs-indexed panel + inventory / chunks / rebuild API."""
from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import patch

import pandas as pd


def _project(client) -> str:
    r = client.post("/api/projects", json={"name": "RAG Docs Panel Test"})
    assert r.status_code in (200, 201), r.text
    return r.json()["id"]


def _write_fake_corpus(root: Path, *, doc_id: str = "abc123def456") -> Path:
    """Minimal PortableRAG-shaped corpus: manifest + sources + chunks.parquet."""
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


def test_rag_page_renders_docs_indexed_panel(
    client, tmp_path: Path, monkeypatch
) -> None:
    pid = _project(client)
    corpus = tmp_path / "corpora" / pid
    _write_fake_corpus(corpus)

    monkeypatch.setattr(
        "finetune_studio.webui.routes.project_rag.corpus_dir",
        lambda p: corpus if p == pid else tmp_path / "corpora" / p,
    )

    r = client.get(f"/projects/{pid}/rag")
    assert r.status_code == 200, r.text
    body = r.text
    assert 'id="rag-docs-panel"' in body
    assert 'id="rag-docs-table"' in body
    for col in ("Doc name", "Type", "Chunks", "Status", "Last indexed", "Actions"):
        assert f">{col}</th>" in body
    assert "manual_notes.txt" in body
    assert "Rebuild" in body
    assert "View chunks" in body
    assert "rebuildDoc" in body
    assert "viewDocChunks" in body
    assert 'id="rag-chunks-modal"' in body
    assert "1 document" in body
    assert "2 chunk" in body


def test_rag_page_empty_state(client) -> None:
    pid = _project(client)
    with patch(
        "finetune_studio.webui.routes.project_rag.list_indexed_docs",
        return_value=[],
    ):
        r = client.get(f"/projects/{pid}/rag")
    assert r.status_code == 200
    assert "No documents indexed yet" in r.text
    assert 'id="rag-docs-panel"' in r.text


def test_list_indexed_docs_helper(tmp_path: Path, monkeypatch) -> None:
    from finetune_studio.webui.routes import project_rag as pr

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
        "finetune_studio.webui.routes.project_rag.corpus_dir",
        lambda p: corpus if p == pid else tmp_path / "missing" / p,
    )
    r = client.get(f"/api/projects/{pid}/rag/docs")
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
        "finetune_studio.webui.routes.project_rag.corpus_dir",
        lambda p: corpus if p == pid else tmp_path / "missing" / p,
    )
    r = client.get(f"/api/projects/{pid}/rag/docs/{doc_id}/chunks")
    assert r.status_code == 200, r.text
    data = r.json()
    assert data["count"] == 2
    assert data["chunks"][0]["chunk_id"] == f"{doc_id}_0"
    assert len(data["chunks"][0]["preview"]) <= 120
    assert data["chunks"][0]["preview"].startswith("Hello")


def test_rebuild_endpoint_mocked(client, tmp_path: Path, monkeypatch) -> None:
    pid = _project(client)
    files_dir = tmp_path / "files"
    files_dir.mkdir(parents=True)
    (files_dir / "a.txt").write_text("alpha beta gamma", encoding="utf-8")

    corpus = tmp_path / "corpora" / pid
    _write_fake_corpus(corpus)

    monkeypatch.setattr(
        "finetune_studio.webui.routes.project_rag.corpus_dir",
        lambda p: corpus if p == pid else tmp_path / "corpora" / p,
    )
    monkeypatch.setattr(
        "finetune_studio.webui.routes.project_rag.project_files_dir",
        lambda p: files_dir if p == pid else tmp_path / "nofiles",
    )

    class _FakeRAG:
        def __init__(self, d: Path) -> None:
            self.dir = d

        def build_from_directory(self, **kwargs: object) -> dict:
            # Re-write corpus so list_indexed_docs still works after "rebuild"
            _write_fake_corpus(self.dir)
            return {"documents": 1, "chunks": 2, "skipped": 0}

    monkeypatch.setattr(
        "finetune_studio.data.rag_portable.PortableRAG",
        _FakeRAG,
    )

    r = client.post(
        f"/api/projects/{pid}/rag/rebuild",
        json={"doc_id": "abc123def456", "reset": True},
    )
    assert r.status_code == 200, r.text
    data = r.json()
    assert data["ok"] is True
    assert data["doc_id"] == "abc123def456"
    assert data["document_count"] == 1
    assert data["chunk_count"] == 2


def test_docs_api_404_unknown_project(client) -> None:
    r = client.get("/api/projects/doesnotexist/rag/docs")
    assert r.status_code == 404
