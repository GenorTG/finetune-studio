"""BrainIndex build must register/update brain_indexes for chat attachments."""
from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

from finetune_studio import db


def _project(client) -> str:
    r = client.post("/api/projects", json={"name": "RAG Build Registration"})
    assert r.status_code in (200, 201), r.text
    return r.json()["id"]


def _patch_build_env(pid: str) -> Path:
    """One parsed source in the project; returns the project's corpus dir."""
    from finetune_studio.data.fs.paths import brain_index_dir
    from tests.knowledge_helpers import add_source

    add_source(pid, "note.txt", "hello corpus, one sentence to index")
    return brain_index_dir(pid)


def test_brain_build_registers_project_rags_row(
    client, tmp_path: Path, monkeypatch
) -> None:
    """Canonical POST …/brain/build writes a brain_indexes row at the corpus dir."""
    pid = _project(client)
    corpus = _patch_build_env(pid)

    mock_rag = MagicMock()
    mock_rag.build_from_documents.return_value = {
        "documents": 3,
        "chunks": 7,
        "vector_dim": 384,
    }

    with patch(
        "finetune_studio.smart_brain.BrainIndex",
        return_value=mock_rag,
    ):
        r = client.post(
            f"/api/projects/{pid}/brain/build",
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

    rags = db.list_brain_indexes(pid)
    assert len(rags) == 1
    assert Path(rags[0]["store_path"]).resolve() == corpus.resolve()
    assert int(rags[0]["doc_count"]) == 3
    assert int(rags[0]["chunk_count"]) == 7
    assert rags[0]["status"] == "ready"
    assert rags[0]["last_build_status"] == "ok"


def test_brain_build_updates_existing_row_same_corpus(
    client, tmp_path: Path, monkeypatch
) -> None:
    """Second build updates the same brain_indexes row (no duplicate)."""
    pid = _project(client)
    corpus = _patch_build_env(pid)
    corpus.mkdir(parents=True)

    first = db.ensure_brain_index(
        pid, str(corpus), name="seed", doc_count=1, chunk_count=2
    )
    mock_rag = MagicMock()
    mock_rag.build_from_documents.return_value = {
        "documents": 9,
        "chunks": 42,
    }
    with patch(
        "finetune_studio.smart_brain.BrainIndex",
        return_value=mock_rag,
    ):
        r = client.post(
            f"/api/projects/{pid}/brain/build",
            json={"reset": False},
        )
    assert r.status_code == 200, r.text
    rags = db.list_brain_indexes(pid)
    assert len(rags) == 1
    assert rags[0]["id"] == first["id"]
    assert int(rags[0]["doc_count"]) == 9
    assert int(rags[0]["chunk_count"]) == 42


def test_ensure_portable_rag_helper_idempotent(temp_db, tmp_path: Path) -> None:
    proj = db.create_project(name="ensure-helper")
    pid = proj["id"]
    store = str((tmp_path / "corpus").resolve())
    a = db.ensure_brain_index(pid, store, name="C", doc_count=1, chunk_count=2)
    b = db.ensure_brain_index(pid, store, name="C", doc_count=5, chunk_count=10)
    assert a["id"] == b["id"]
    assert int(b["doc_count"]) == 5
    assert int(b["chunk_count"]) == 10
    assert len(db.list_brain_indexes(pid)) == 1
