"""RAG rebuild replaces stale sources; list_sources follows manifest/chunks."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from finetune_studio.smart_brain.schema import EmbeddingModelInfo
from finetune_studio.smart_brain.store import BrainIndex
from finetune_studio.webui.routes.brain_documents import list_indexed_docs


def _fake_get_embedder(name: str = "fake-deterministic", device: str = "cpu"):
    dim = 32

    def encode(texts: list[str] | str) -> np.ndarray:
        items = [texts] if isinstance(texts, str) else list(texts)
        out = np.zeros((len(items), dim), dtype=np.float32)
        for i, text in enumerate(items):
            out[i, hash(text) % dim] = 1.0
            n = float(np.linalg.norm(out[i]))
            if n > 0:
                out[i] /= n
        return out[0] if isinstance(texts, str) else out

    info = EmbeddingModelInfo(
        name=name or "fake-deterministic",
        dim=dim,
        normalize=True,
        distance="cosine",
        cached_at="1970-01-01T00:00:00Z",
    )
    return encode, info


@pytest.fixture()
def patched_embedder(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        "finetune_studio.smart_brain.store.get_embedder",
        _fake_get_embedder,
    )


def test_rebuild_clears_stale_sources_and_list_sources(
    tmp_path: Path, patched_embedder: None
) -> None:
    """A plain-directory build replaces the corpus: orphans from older builds go, a removed file leaves the index.
    (Project indexes are built from parsed sources: tests/test_knowledge_index.py.)"""
    src = tmp_path / "docs"
    src.mkdir()
    (src / "helios_architecture.txt").write_text("Helios arch doc " * 20, encoding="utf-8")
    (src / "ops_runbook.txt").write_text("Ops runbook body " * 20, encoding="utf-8")
    corpus = tmp_path / "corpus"

    rag = BrainIndex(corpus)
    # Plant stale orphans that older builds left behind (chunk + opaque).
    (corpus / "sources").mkdir(parents=True, exist_ok=True)
    (corpus / "sources" / "deadbeef0001.txt").write_text("orphan chunk", encoding="utf-8")
    (corpus / "sources" / "deadbeef0002.txt").write_text("orphan opaque", encoding="utf-8")

    stats1 = rag.build_from_directory(src, name="rebuild-test", embedder="fake-deterministic", chunk_size=80,
                                      overlap=10, extensions=[".txt"])
    assert stats1["documents"] == 2
    source_files = sorted(p.name for p in (corpus / "sources").glob("*.txt"))
    assert "deadbeef0001.txt" not in source_files and "deadbeef0002.txt" not in source_files
    assert len(source_files) == 2
    listed = rag.list_sources()
    assert {s["filename"] for s in listed} == {"helios_architecture.txt", "ops_runbook.txt"}
    assert {s["id"] for s in listed} == {p.stem for p in (corpus / "sources").glob("*.txt")}

    (src / "ops_runbook.txt").unlink()
    stats2 = rag.build_from_directory(src, name="rebuild-test", embedder="fake-deterministic", chunk_size=80,
                                      overlap=10, extensions=[".txt"])
    assert stats2["documents"] == 1
    assert len(list((corpus / "sources").glob("*.txt"))) == 1
    assert len(rag.list_sources()) == 1


def test_list_indexed_docs_ignores_orphan_sources(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import pandas as pd

    pid = "orphanpid01"
    corpus = tmp_path / "corpora" / pid
    (corpus / "sources").mkdir(parents=True)
    (corpus / "sources" / "orphanonly.txt").write_text("stale", encoding="utf-8")
    (corpus / "sources" / "keepme.txt").write_text("keep", encoding="utf-8")

    df = pd.DataFrame([
        {
            "id": "keepme_0",
            "document_id": "keepme",
            "chunk_index": 0,
            "source": "/tmp/files/sha/parsed.txt",
            "filename": "report.docx (parsed)",
            "text": "keep",
        }
    ])
    df.to_parquet(corpus / "chunks.parquet", index=False)
    (corpus / "manifest.json").write_text(
        json.dumps({
            "updated_at": 1_700_000_000.0,
            "documents": 1,
            "chunks": 1,
            "extra": {
                "documents_meta": [{
                    "document_id": "keepme",
                    "filename": "report.docx (parsed)",
                    "source": "/tmp/files/sha/parsed.txt",
                }],
            },
        }),
        encoding="utf-8",
    )

    monkeypatch.setattr(
        "finetune_studio.webui.routes.brain_documents.corpus_dir",
        lambda p: corpus if p == pid else tmp_path / "other" / p,
    )
    docs = list_indexed_docs(pid)
    assert len(docs) == 1
    assert docs[0]["id"] == "keepme"
    assert docs[0]["name"] == "report.docx (parsed)"


def test_list_sources_ignores_orphan_txt_without_manifest_meta(
    tmp_path: Path, patched_embedder: None
) -> None:
    src = tmp_path / "files"
    src.mkdir(parents=True)
    corpus = tmp_path / "corpus"
    (src / "only.txt").write_text("hello world content for chunking " * 10, encoding="utf-8")

    rag = BrainIndex(corpus)
    rag.build_from_directory(
        src,
        name="meta",
        embedder="fake-deterministic",
        chunk_size=40,
        overlap=5,
        extensions=[".txt"],
    )
    (corpus / "sources" / "ghostorphan.txt").write_text("ghost", encoding="utf-8")

    listed = rag.list_sources()
    assert all(s["id"] != "ghostorphan" for s in listed)
    assert len(listed) == 1
