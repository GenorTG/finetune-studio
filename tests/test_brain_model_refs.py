"""Smart Brain model references: what a build, a re-embed and an export do with the embedder / reranker names.

Found by the Quillmoor benchmark (2026-10-29): after "Re-embed only" with another embedder the index could no longer
be exported with its models (409 "does not reference a shared embedder"); a reranker chosen in the search settings
was never bundled (the package shipped with the reranker off and said nothing); every rebuild reset the chosen
reranker to the default one.
"""
from __future__ import annotations

import json
import tarfile
from pathlib import Path

import numpy as np
import pytest

from finetune_studio.data import shared_models as sm
from finetune_studio.smart_brain.constants import DEFAULT_RERANKER
from finetune_studio.smart_brain.schema import EmbeddingModelInfo
from finetune_studio.smart_brain.server_package import build_package
from finetune_studio.smart_brain.store import BrainIndex


def _fake_get_embedder(name: str = "fake", device: str = "cpu"):
    def encode(texts):
        items = [texts] if isinstance(texts, str) else list(texts)
        out = np.zeros((len(items), 8), dtype=np.float32)
        for i, t in enumerate(items):
            out[i, len(t) % 8] = 1.0
        return out[0] if isinstance(texts, str) else out
    return encode, EmbeddingModelInfo(name=name, dim=8, normalize=True, distance="cosine", cached_at="x")


@pytest.fixture()
def store(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> dict[str, Path]:
    """A fake shared model store: register() makes a model dir per name, resolve() finds it again."""
    dirs: dict[str, Path] = {}
    calls: list[str] = []

    def register(model_name: str, kind: str, src_dir=None, prefer_cached: bool = True):
        calls.append(f"{kind}:{model_name}")
        short = model_name.replace("/", "__") + "@fake"
        d = tmp_path / "shared" / kind / short
        d.mkdir(parents=True, exist_ok=True)
        (d / "config.json").write_text("{}")
        dirs[short] = d
        return sm.ModelRef(name=model_name, kind=kind, short_id=short, path=str(d))

    def resolve(short_id: str, kind: str) -> Path:
        if short_id not in dirs:
            raise ValueError(short_id)
        return dirs[short_id]

    monkeypatch.setattr(sm, "register", register)
    monkeypatch.setattr(sm, "resolve", resolve)
    monkeypatch.setattr("finetune_studio.smart_brain.store.get_embedder", _fake_get_embedder)
    return {"calls": calls}  # type: ignore[dict-item]


def _build(tmp_path: Path, corpus: Path) -> BrainIndex:
    src = tmp_path / "docs"
    src.mkdir(exist_ok=True)
    (src / "a.txt").write_text("The pump PMP-4471-B is serviced every 6,500 operating hours. " * 4)
    (src / "b.txt").write_text("Contract QC-2021-047 is worth 839,500 EUR. " * 4)
    idx = BrainIndex(corpus)
    idx.build_from_directory(src, name="t", embedder="org/first-embedder", chunk_size=80, overlap=10, extensions=[".txt"])
    return idx


def _manifest(corpus: Path) -> dict:
    return json.loads((corpus / "manifest.json").read_text())


def test_reembed_with_another_embedder_can_still_be_exported_with_models(tmp_path: Path, store) -> None:
    corpus = tmp_path / "corpus"
    idx = _build(tmp_path, corpus)
    idx.rebuild_vectors(embedder="org/second-embedder")
    m = _manifest(corpus)
    assert m["embedding_model"]["name"] == "shared:embedder:org__second-embedder@fake"
    assert m["rag_settings"]["embedder"] == m["embedding_model"]["name"]
    assert m["extra"]["shared_model_paths"]["embedder"].endswith("org__second-embedder@fake")
    out = build_package(corpus, tmp_path / "p.tar", fmt="tar", include_models=True, encrypt=False)
    with tarfile.open(out.path) as tf:
        names = tf.getnames()
    assert any(n.endswith("corpus/embedder/config.json") for n in names)


def test_reembed_keeps_a_shared_ref_it_was_given(tmp_path: Path, store) -> None:
    corpus = tmp_path / "corpus"
    idx = _build(tmp_path, corpus)
    ref = _manifest(corpus)["embedding_model"]["name"]
    assert ref == "shared:embedder:org__first-embedder@fake"
    idx.rebuild_vectors()                       # no embedder = the one the settings name (a shared ref)
    assert _manifest(corpus)["embedding_model"]["name"] == ref


def test_rebuild_keeps_the_reranker_the_settings_chose(tmp_path: Path, store) -> None:
    corpus = tmp_path / "corpus"
    _build(tmp_path, corpus)
    assert _manifest(corpus)["rag_settings"]["reranker"] == f"shared:reranker:{DEFAULT_RERANKER.replace('/', '__')}@fake"
    m = _manifest(corpus)
    m["rag_settings"]["reranker"] = "BAAI/bge-reranker-v2-m3"       # what POST /brain/settings saves
    (corpus / "manifest.json").write_text(json.dumps(m))
    _build(tmp_path, corpus)
    assert _manifest(corpus)["rag_settings"]["reranker"] == "shared:reranker:BAAI__bge-reranker-v2-m3@fake"


def test_export_bundles_a_reranker_saved_by_name(tmp_path: Path, store) -> None:
    corpus = tmp_path / "corpus"
    _build(tmp_path, corpus)
    m = _manifest(corpus)
    m["rag_settings"]["reranker"] = "BAAI/bge-reranker-v2-m3"
    (corpus / "manifest.json").write_text(json.dumps(m))
    out = build_package(corpus, tmp_path / "p.tar", fmt="tar", include_models=True, encrypt=False)
    assert out.layers["reranker"] is True
    with tarfile.open(out.path) as tf:
        assert any(n.endswith("corpus/reranker/config.json") for n in tf.getnames())
    assert "reranker:BAAI/bge-reranker-v2-m3" in store["calls"]


def test_export_says_when_the_chosen_reranker_cannot_be_fetched(tmp_path: Path, store, monkeypatch) -> None:
    corpus = tmp_path / "corpus"
    _build(tmp_path, corpus)
    m = _manifest(corpus)
    m["rag_settings"]["reranker"] = "nobody/no-such-reranker"
    (corpus / "manifest.json").write_text(json.dumps(m))

    def boom(*a, **k):
        raise OSError("404 not found")

    monkeypatch.setattr(sm, "register", boom)
    with pytest.raises(FileNotFoundError, match="could not be bundled"):
        build_package(corpus, tmp_path / "p.tar", fmt="tar", include_models=True, encrypt=False)
