"""RAG rebuild replaces stale sources; list_sources follows manifest/chunks."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from finetune_studio.data.rag_portable.schema import EmbeddingModelInfo
from finetune_studio.data.rag_portable.store import PortableRAG
from finetune_studio.webui.routes.project_rag import list_indexed_docs


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
        "finetune_studio.data.rag_portable.store.get_embedder",
        _fake_get_embedder,
    )


def _sha_dir(root: Path, sha: str, original: str, body: str) -> Path:
    d = root / sha
    d.mkdir(parents=True)
    (d / "metadata.json").write_text(
        json.dumps({"original_filename": original}),
        encoding="utf-8",
    )
    (d / "parsed.txt").write_text(body, encoding="utf-8")
    chunks = d / "chunks"
    chunks.mkdir()
    (chunks / "0000.txt").write_text(body[:40], encoding="utf-8")
    return d


def test_rebuild_clears_stale_sources_and_list_sources(
    tmp_path: Path, patched_embedder: None
) -> None:
    src = tmp_path / "files"
    corpus = tmp_path / "corpus"
    _sha_dir(src, "aaa111bbb222", "helios_architecture.docx", "Helios arch doc " * 20)
    _sha_dir(src, "ccc333ddd444", "ops_runbook.md", "Ops runbook body " * 20)

    rag = PortableRAG(corpus)
    # Plant stale orphans that older builds left behind (chunk + opaque).
    (corpus / "sources").mkdir(parents=True, exist_ok=True)
    (corpus / "sources" / "deadbeef0001.txt").write_text("orphan chunk", encoding="utf-8")
    (corpus / "sources" / "deadbeef0002.txt").write_text("orphan opaque", encoding="utf-8")

    stats1 = rag.build_from_directory(
        src,
        name="rebuild-test",
        embedder="fake-deterministic",
        chunk_size=80,
        overlap=10,
        extensions=[".txt"],
    )
    assert stats1["documents"] == 2

    source_files = sorted(p.name for p in (corpus / "sources").glob("*.txt"))
    assert "deadbeef0001.txt" not in source_files
    assert "deadbeef0002.txt" not in source_files
    assert len(source_files) == 2

    listed = rag.list_sources()
    assert len(listed) == 2
    names = {s["filename"] for s in listed}
    assert any("helios_architecture.docx" in n for n in names)
    assert any("ops_runbook.md" in n for n in names)
    assert not any("chunk 0" in n for n in names)
    assert {s["id"] for s in listed} == {p.stem for p in (corpus / "sources").glob("*.txt")}

    # Second rebuild with one source removed — corpus shrinks; no stale leftovers.
    import shutil

    shutil.rmtree(src / "ccc333ddd444")
    stats2 = rag.build_from_directory(
        src,
        name="rebuild-test",
        embedder="fake-deterministic",
        chunk_size=80,
        overlap=10,
        extensions=[".txt"],
    )
    assert stats2["documents"] == 1
    assert len(list((corpus / "sources").glob("*.txt"))) == 1
    assert len(rag.list_sources()) == 1


def test_ingest_dedups_chunks_vs_parsed(
    tmp_path: Path, patched_embedder: None
) -> None:
    src = tmp_path / "files"
    corpus = tmp_path / "corpus"
    _sha_dir(
        src,
        "eeeeffff0001",
        "helios_architecture.docx",
        "Single logical document " * 30,
    )

    rag = PortableRAG(corpus)
    stats = rag.build_from_directory(
        src,
        name="dedup",
        embedder="fake-deterministic",
        chunk_size=60,
        overlap=8,
        extensions=[".txt"],
    )
    assert stats["documents"] == 1
    listed = rag.list_sources()
    assert len(listed) == 1
    assert "parsed" in listed[0]["filename"]
    assert "chunk" not in listed[0]["filename"].lower()


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
        "finetune_studio.webui.routes.project_rag.corpus_dir",
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

    rag = PortableRAG(corpus)
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
