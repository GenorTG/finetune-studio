"""PortableRAG build must register/update project_rags for chat attachments."""
from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

from finetune_studio import db


def _project(client) -> str:
    r = client.post("/api/projects", json={"name": "RAG Build Registration"})
    assert r.status_code in (200, 201), r.text
    return r.json()["id"]


def _patch_build_env(monkeypatch, tmp_path: Path, pid: str) -> Path:
    """Point corpus + project files under tmp_path; return corpus dir."""
    corpus = tmp_path / "rag_corpora" / pid
    home = tmp_path / "home"
    files = home / ".finetune-studio" / "projects" / pid / "files"
    files.mkdir(parents=True)
    (files / "note.txt").write_text("hello corpus", encoding="utf-8")
    monkeypatch.setattr(
        "finetune_studio.webui.routes.rag._corpus_dir",
        lambda p: corpus if p == pid else tmp_path / "other" / p,
    )
    monkeypatch.setattr(
        "finetune_studio.webui.routes.rag.Path.home",
        staticmethod(lambda: home),
    )
    return corpus


def test_rag_build_registers_project_rags_row(
    client, tmp_path: Path, monkeypatch
) -> None:
    """Canonical POST …/rag/build writes a project_rags row at the corpus dir."""
    pid = _project(client)
    corpus = _patch_build_env(monkeypatch, tmp_path, pid)

    mock_rag = MagicMock()
    mock_rag.build_from_directory.return_value = {
        "documents": 3,
        "chunks": 7,
        "vector_dim": 384,
    }

    with patch(
        "finetune_studio.data.rag_portable.PortableRAG",
        return_value=mock_rag,
    ):
        r = client.post(
            f"/api/projects/{pid}/rag/build",
            json={
                "chunk_size": 400,
                "overlap": 80,
                "embedder": "sentence-transformers/all-MiniLM-L6-v2",
                "reset": True,
            },
        )
    assert r.status_code == 200, r.text
    body = r.json()
    assert body.get("ok") is True
    # Sync build must not claim an in-flight job after the handler returns.
    assert body.get("building") is False
    assert body.get("documents") == 3
    assert body.get("chunks") == 7

    rags = db.list_rags(pid)
    assert len(rags) == 1
    assert Path(rags[0]["store_path"]).resolve() == corpus.resolve()
    assert int(rags[0]["doc_count"]) == 3
    assert int(rags[0]["chunk_count"]) == 7
    assert rags[0]["status"] == "ready"
    assert rags[0]["last_build_status"] == "ok"


def test_rag_build_updates_existing_row_same_corpus(
    client, tmp_path: Path, monkeypatch
) -> None:
    """Second build updates the same project_rags row (no duplicate)."""
    pid = _project(client)
    corpus = _patch_build_env(monkeypatch, tmp_path, pid)
    corpus.mkdir(parents=True)

    first = db.ensure_portable_rag(
        pid, str(corpus), name="seed", doc_count=1, chunk_count=2
    )
    mock_rag = MagicMock()
    mock_rag.build_from_directory.return_value = {
        "documents": 9,
        "chunks": 42,
    }
    with patch(
        "finetune_studio.data.rag_portable.PortableRAG",
        return_value=mock_rag,
    ):
        r = client.post(
            f"/api/projects/{pid}/rag/build",
            json={"reset": False},
        )
    assert r.status_code == 200, r.text
    rags = db.list_rags(pid)
    assert len(rags) == 1
    assert rags[0]["id"] == first["id"]
    assert int(rags[0]["doc_count"]) == 9
    assert int(rags[0]["chunk_count"]) == 42


def test_ensure_portable_rag_helper_idempotent(tmp_path: Path) -> None:
    proj = db.create_project(name="ensure-helper")
    pid = proj["id"]
    store = str((tmp_path / "corpus").resolve())
    a = db.ensure_portable_rag(pid, store, name="C", doc_count=1, chunk_count=2)
    b = db.ensure_portable_rag(pid, store, name="C", doc_count=5, chunk_count=10)
    assert a["id"] == b["id"]
    assert int(b["doc_count"]) == 5
    assert int(b["chunk_count"]) == 10
    assert len(db.list_rags(pid)) == 1
