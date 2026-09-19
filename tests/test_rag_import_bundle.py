"""Tests for PortableRAG.import_bundle + the /rag/import upload route.

Round-trips a minimal-but-real corpus (tiny parquet/npy payloads so the
post-import ``load()`` validation actually runs) through export → import,
plus refusal/overwrite and manifest-validation paths.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from finetune_studio.data.rag_portable import (
    EMBEDDER_LOCAL_PREFIX,
    RERANKER_LOCAL_PREFIX,
    ChunkSettings,
    EmbeddingModelInfo,
    Manifest,
    PortableRAG,
    RagSettings,
    write_json,
)
from finetune_studio.data.rag_portable.store import (
    EMBEDDER_LOCAL_PREFIX as _ELP,  # noqa: F401
)
from finetune_studio.data.rag_portable.store import (
    RERANKER_LOCAL_PREFIX as _RLP,  # noqa: F401
)


def _write_minimal_corpus(corpus: Path, *, with_local_models: bool = False) -> Manifest:
    """Local fixture (mirrors test_rag_export_isolation, self-contained)."""
    corpus.mkdir(parents=True, exist_ok=True)
    (corpus / "sources").mkdir(exist_ok=True)
    (corpus / "sources" / "doc1.txt").write_text("hello world " * 20, encoding="utf-8")
    (corpus / "vectors.idx.json").write_text("{}", encoding="utf-8")

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
        name="import-test",
        version="2",
        created_at=1.0,
        updated_at=2.0,
        embedding_model=EmbeddingModelInfo(name=emb_ref, dim=4),
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


def _write_real_payloads(corpus: Path, dim: int = 4) -> None:
    """Write load()-compatible tiny parquet/vectors/bm25 (placeholders die)."""
    pd = pytest.importorskip("pandas")
    np = pytest.importorskip("numpy")

    chunks = pd.DataFrame(
        {
            "id": ["doc1#0"],
            "doc_id": ["doc1"],
            "chunk_idx": [0],
            "text": ["hello world " * 20],
        }
    )
    chunks.to_parquet(corpus / "chunks.parquet", index=False)
    vecs = np.zeros((1, dim), dtype=np.float32)
    vecs[0, 0] = 1.0
    np.save(corpus / "vectors.npy", vecs)
    (corpus / "bm25.json").write_text(
        json.dumps({
            "terms": {},
            "doc_lens": [20],
            "df": {},
            "avgdl": 20.0,
            "doc_count": 1,
        }),
        encoding="utf-8",
    )


def test_import_round_trip_tar(tmp_path: Path, monkeypatch) -> None:
    """Export → import into a fresh dir; parquet/chunks survive intact."""
    src = tmp_path / "src"
    _write_minimal_corpus(src, with_local_models=False)
    _write_real_payloads(src)
    rag_src = PortableRAG(src)
    archive = tmp_path / "bundle.tar"
    rag_src.export_bundle(out_path=archive, fmt="tar", name="bundle")

    dst = tmp_path / "dst"
    _stub_embedder(monkeypatch)
    stats = PortableRAG(dst).import_bundle(archive)

    assert stats["documents"] == 1
    assert stats["chunks"] == 1
    pd = pytest.importorskip("pandas")
    chunks = pd.read_parquet(dst / "chunks.parquet")
    assert len(chunks) == 1
    assert (dst / "sources" / "doc1.txt").exists()


def _stub_embedder(monkeypatch, dim: int = 4) -> None:
    """make load()'s embedder resolution succeed without a real model."""
    np = pytest.importorskip("numpy")
    import finetune_studio.data.rag_portable.store as store_mod

    def fake_get_embedder(name: str):
        def encode(texts):
            return np.zeros((len(texts), dim), dtype="float32")

        class Info:
            pass

        info = Info()
        info.dim = dim
        return encode, info

    monkeypatch.setattr(store_mod, "get_embedder", fake_get_embedder)


def test_import_local_models_repointed(tmp_path: Path, monkeypatch) -> None:
    """include_models bundle: after import, embedder/reranker refs target dst."""
    src = tmp_path / "src"
    _write_minimal_corpus(src, with_local_models=True)
    _write_real_payloads(src)
    PortableRAG(src).export_bundle(
        out_path=tmp_path / "fat.tar", fmt="tar", name="fat", include_models=True
    )
    dst = tmp_path / "dst"
    _stub_embedder(monkeypatch)
    PortableRAG(dst).import_bundle(tmp_path / "fat.tar")
    manifest = json.loads((dst / "manifest.json").read_text(encoding="utf-8"))
    emb = manifest["embedding_model"]["name"]
    rr = manifest["rag_settings"]["reranker"]
    assert emb.startswith(EMBEDDER_LOCAL_PREFIX) and str(dst) in emb
    assert rr.startswith(RERANKER_LOCAL_PREFIX) and str(dst) in rr


def test_import_refuses_existing_corpus_without_overwrite(tmp_path: Path, monkeypatch) -> None:
    src = tmp_path / "src"
    _write_minimal_corpus(src, with_local_models=False)
    _write_real_payloads(src)
    archive = tmp_path / "b.tar"
    PortableRAG(src).export_bundle(out_path=archive, fmt="tar", name="b")

    dst = tmp_path / "dst"
    _write_minimal_corpus(dst, with_local_models=False)
    _write_real_payloads(dst)
    # Marker file that only the pre-existing corpus has — import must NOT
    # touch it until overwrite=True.
    (dst / "sentinel.txt").write_text("existing", encoding="utf-8")
    _stub_embedder(monkeypatch)
    with pytest.raises(FileExistsError):
        PortableRAG(dst).import_bundle(archive)
    # Nothing was clobbered.
    assert (dst / "sentinel.txt").read_text(encoding="utf-8") == "existing"

    # With overwrite=True the old tree is replaced.
    stats = PortableRAG(dst).import_bundle(archive, overwrite=True)
    assert stats["documents"] == 1
    assert not (dst / "sentinel.txt").exists()


def test_import_rejects_missing_archive_and_bad_manifest(tmp_path: Path) -> None:
    dst = tmp_path / "dst"
    rag = PortableRAG(dst)
    with pytest.raises(FileNotFoundError):
        rag.import_bundle(tmp_path / "nope.tar")

    bad = tmp_path / "bad.zip"
    import zipfile

    with zipfile.ZipFile(bad, "w") as zf:
        zf.writestr("random.txt", "not a bundle")
    with pytest.raises(ValueError):
        rag.import_bundle(bad)


def test_import_rejects_path_traversal(tmp_path: Path) -> None:
    """A bundle with ../ escape entries must be refused, not extracted."""
    import tarfile
    import zipfile

    evil = tmp_path / "evil.zip"
    with zipfile.ZipFile(evil, "w") as zf:
        zf.writestr("../../pwned.txt", "boom")
    with pytest.raises(ValueError, match="unsafe path"):
        PortableRAG(tmp_path / "newdir").import_bundle(evil)
    assert not (tmp_path / "pwned.txt").exists()

    evil_tar = tmp_path / "evil.tar"
    with tarfile.open(evil_tar, "w") as tar:
        import io

        data = b"boom"
        info = tarfile.TarInfo(name="../pwned2.txt")
        info.size = len(data)
        tar.addfile(info, io.BytesIO(data))
    with pytest.raises((tarfile.TarError, ValueError)):
        PortableRAG(tmp_path / "newdir2").import_bundle(evil_tar)
    assert not (tmp_path / "pwned2.txt").exists()


def test_import_route_accepts_tar_gz_suffix(tmp_path: Path, monkeypatch) -> None:
    """The upload route must accept 'x.tar.gz' — Path.suffix sees '.gz' only."""
    import io
    
    from fastapi import FastAPI
    from fastapi.testclient import TestClient

    import finetune_studio.webui.routes.rag as rag_routes

    monkeypatch.setattr(rag_routes, "_CORPORA", tmp_path / "rag_corpora")

    app = FastAPI()
    app.include_router(rag_routes.router)
    client = TestClient(app)

    # minimal valid bundle built via the store's own export path; with local
    # models so |load()| validation can resolve them without a shared store
    src = tmp_path / "srccorpus"
    _write_minimal_corpus(src, with_local_models=True)
    _write_real_payloads(src)
    archive = PortableRAG(src).export_bundle(out_path=None, fmt="tar.gz", include_models=True)
    data = Path(archive).read_bytes()
    _stub_embedder(monkeypatch)

    pid = "routecheck01"
    r = client.post(
        f"/{pid}/rag/import?overwrite=false",
        files={"file": ("vael-bundle.tar.gz", io.BytesIO(data), "application/gzip")},
    )
    assert r.status_code == 200, r.text
    body = r.json()
    assert body.get("documents", 0) >= 1 or body.get("chunks", 0) >= 1
    assert (tmp_path / "rag_corpora" / pid / "manifest.json").exists()
