"""RAG bundle export isolation — include_models must not leak or mutate corpus."""
from __future__ import annotations

import json
import tarfile
from pathlib import Path

from finetune_studio.data.rag_portable.constants import (
    EMBEDDER_LOCAL_PREFIX,
    RERANKER_LOCAL_PREFIX,
)
from finetune_studio.data.rag_portable.io import write_json
from finetune_studio.data.rag_portable.schema import (
    ChunkSettings,
    EmbeddingModelInfo,
    Manifest,
    RagSettings,
)
from finetune_studio.data.rag_portable.store import PortableRAG


def _write_minimal_corpus(corpus: Path, *, with_local_models: bool = False) -> Manifest:
    """Create a tiny PortableRAG-shaped corpus with shared refs (+ optional staged models)."""
    corpus.mkdir(parents=True, exist_ok=True)
    (corpus / "sources").mkdir(exist_ok=True)
    (corpus / "sources" / "doc1.txt").write_text("hello world " * 20, encoding="utf-8")
    (corpus / "chunks.parquet").write_bytes(b"PARQUET_PLACEHOLDER")
    (corpus / "vectors.npy").write_bytes(b"NUMPY_PLACEHOLDER")
    (corpus / "vectors.idx.json").write_text("{}", encoding="utf-8")
    (corpus / "bm25.json").write_text("{}", encoding="utf-8")

    shared_paths = {
        "embedder": "/tmp/shared/embedders/demo-embed@abc",
        "reranker": "/tmp/shared/rerankers/demo-rerank@def",
    }
    emb_ref = "shared:embedder:demo-embed@abc"
    rr_ref = "shared:reranker:demo-rerank@def"
    if with_local_models:
        emb_dir = corpus / "embedder"
        rr_dir = corpus / "reranker"
        emb_dir.mkdir()
        rr_dir.mkdir()
        (emb_dir / "model.safetensors").write_bytes(b"E" * 4096)
        (rr_dir / "model.safetensors").write_bytes(b"R" * 4096)
        emb_ref = f"{EMBEDDER_LOCAL_PREFIX}{emb_dir.resolve()}"
        rr_ref = f"{RERANKER_LOCAL_PREFIX}{rr_dir.resolve()}"

    m = Manifest(
        name="iso-test",
        version="2",
        created_at=1.0,
        updated_at=2.0,
        embedding_model=EmbeddingModelInfo(name=emb_ref, dim=8),
        chunk_settings=ChunkSettings(size=400, overlap=80),
        rag_settings=RagSettings(
            embedder=emb_ref,
            reranker=rr_ref,
            rerank_enabled=True,
            hybrid_enabled=True,
        ),
        documents=1,
        chunks=1,
        extra={"shared_model_paths": shared_paths},
    )
    write_json(corpus / "manifest.json", m.to_json())
    return m


def _archive_names(archive: Path) -> list[str]:
    with tarfile.open(archive, "r:") as tar:
        return [m.name for m in tar.getmembers()]


def _archive_manifest(archive: Path, corpus_name: str) -> dict:
    with tarfile.open(archive, "r:") as tar:
        member = tar.getmember(f"{corpus_name}/manifest.json")
        f = tar.extractfile(member)
        assert f is not None
        return json.loads(f.read().decode("utf-8"))


def test_include_models_false_excludes_staged_embedder(
    tmp_path: Path,
) -> None:
    """Even if a prior with-models export left embedder/ on disk, slim export omits it."""
    corpus = tmp_path / "corpus-a"
    _write_minimal_corpus(corpus, with_local_models=True)
    rag = PortableRAG(corpus)
    before = (corpus / "manifest.json").read_text(encoding="utf-8")
    emb_mtime = (corpus / "embedder" / "model.safetensors").stat().st_mtime_ns

    out = rag.export_bundle(
        out_path=tmp_path / "slim.tar",
        fmt="tar",
        name="slim",
        include_models=False,
    )
    assert out.exists()
    names = _archive_names(out)
    assert any(n.endswith("manifest.json") for n in names)
    assert not any("/embedder/" in n or n.endswith("/embedder") for n in names)
    assert not any("/reranker/" in n or n.endswith("/reranker") for n in names)
    assert not any(n.endswith("model.safetensors") for n in names)

    man = _archive_manifest(out, corpus.name)
    emb = man["embedding_model"]["name"]
    rr = man["rag_settings"]["reranker"]
    assert emb.startswith("shared:embedder:"), emb
    assert rr.startswith("shared:reranker:"), rr
    assert EMBEDDER_LOCAL_PREFIX not in emb
    assert RERANKER_LOCAL_PREFIX not in rr

    # Live corpus untouched
    assert (corpus / "manifest.json").read_text(encoding="utf-8") == before
    assert (corpus / "embedder" / "model.safetensors").exists()
    assert (corpus / "embedder" / "model.safetensors").stat().st_mtime_ns == emb_mtime


def test_include_models_true_stages_without_mutating_live(
    tmp_path: Path, monkeypatch
) -> None:
    """With-models export ships weights from shared paths but does not mutate corpus."""
    corpus = tmp_path / "corpus-b"
    _write_minimal_corpus(corpus, with_local_models=False)
    shared_emb = tmp_path / "shared-emb"
    shared_rr = tmp_path / "shared-rr"
    shared_emb.mkdir()
    shared_rr.mkdir()
    (shared_emb / "weights.bin").write_bytes(b"EMBED" * 200)
    (shared_rr / "weights.bin").write_bytes(b"RERANK" * 200)

    # Point shared_model_paths at our fake shared dirs
    man_path = corpus / "manifest.json"
    data = json.loads(man_path.read_text(encoding="utf-8"))
    data["extra"]["shared_model_paths"] = {
        "embedder": str(shared_emb),
        "reranker": str(shared_rr),
    }
    man_path.write_text(json.dumps(data), encoding="utf-8")

    def _fake_resolve(short_id: str, kind: str) -> Path:
        return shared_emb if kind == "embedder" else shared_rr

    monkeypatch.setattr(
        "finetune_studio.data.rag_portable.store._sm.resolve",
        _fake_resolve,
    )

    rag = PortableRAG(corpus)
    before_manifest = man_path.read_text(encoding="utf-8")
    assert not (corpus / "embedder").exists()

    out = rag.export_bundle(
        out_path=tmp_path / "fat.tar",
        fmt="tar",
        name="fat",
        include_models=True,
    )
    names = _archive_names(out)
    assert any("/embedder/" in n and n.endswith("weights.bin") for n in names)
    assert any("/reranker/" in n and n.endswith("weights.bin") for n in names)

    # Live corpus must remain shared-ref only — no staged dirs written back
    assert not (corpus / "embedder").exists()
    assert not (corpus / "reranker").exists()
    assert man_path.read_text(encoding="utf-8") == before_manifest


def test_slim_after_fat_stays_small(tmp_path: Path, monkeypatch) -> None:
    """Regression: fat then slim must not leave ~model-sized slim archives."""
    corpus = tmp_path / "corpus-c"
    _write_minimal_corpus(corpus, with_local_models=False)
    shared_emb = tmp_path / "shared-emb2"
    shared_rr = tmp_path / "shared-rr2"
    shared_emb.mkdir()
    shared_rr.mkdir()
    big = b"X" * (512 * 1024)
    (shared_emb / "model.safetensors").write_bytes(big)
    (shared_rr / "model.safetensors").write_bytes(big)

    man_path = corpus / "manifest.json"
    data = json.loads(man_path.read_text(encoding="utf-8"))
    data["extra"]["shared_model_paths"] = {
        "embedder": str(shared_emb),
        "reranker": str(shared_rr),
    }
    man_path.write_text(json.dumps(data), encoding="utf-8")

    monkeypatch.setattr(
        "finetune_studio.data.rag_portable.store._sm.resolve",
        lambda short_id, kind: shared_emb if kind == "embedder" else shared_rr,
    )

    rag = PortableRAG(corpus)
    fat = rag.export_bundle(
        out_path=tmp_path / "fat2.tar", fmt="tar", include_models=True
    )
    # Simulate the old bug: leave models on disk after a with-models export
    (corpus / "embedder").mkdir()
    (corpus / "reranker").mkdir()
    (corpus / "embedder" / "model.safetensors").write_bytes(big)
    (corpus / "reranker" / "model.safetensors").write_bytes(big)
    data2 = json.loads(man_path.read_text(encoding="utf-8"))
    data2["embedding_model"]["name"] = (
        f"{EMBEDDER_LOCAL_PREFIX}{(corpus / 'embedder').resolve()}"
    )
    data2["rag_settings"]["embedder"] = data2["embedding_model"]["name"]
    data2["rag_settings"]["reranker"] = (
        f"{RERANKER_LOCAL_PREFIX}{(corpus / 'reranker').resolve()}"
    )
    man_path.write_text(json.dumps(data2), encoding="utf-8")

    slim = rag.export_bundle(
        out_path=tmp_path / "slim2.tar", fmt="tar", include_models=False
    )
    assert fat.stat().st_size > 400_000
    assert slim.stat().st_size < 50_000
    names = _archive_names(slim)
    assert not any("model.safetensors" in n for n in names)
    man = _archive_manifest(slim, corpus.name)
    assert man["embedding_model"]["name"].startswith("shared:embedder:")
