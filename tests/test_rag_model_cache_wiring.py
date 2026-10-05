"""The RAG model cache is used by PortableRAG and released before anything that needs the VRAM."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from finetune_studio.data.rag_portable import model_cache as mc
from finetune_studio.data.rag_portable.schema import EmbeddingModelInfo
from finetune_studio.data.rag_portable.store import PortableRAG

DIM = 16


def _fake_embedder(name: str = "fake-e", device: str = "cpu"):
    def encode(texts):
        items = [texts] if isinstance(texts, str) else list(texts)
        out = np.zeros((len(items), DIM), dtype=np.float32)
        for i, t in enumerate(items):
            out[i, sum(map(ord, t)) % DIM] = 1.0
        return out[0] if isinstance(texts, str) else out

    return encode, EmbeddingModelInfo(
        name=name, dim=DIM, normalize=True, distance="cosine", cached_at="1970-01-01T00:00:00Z",
    )


@pytest.fixture
def constructed(monkeypatch: pytest.MonkeyPatch) -> list[tuple[str, str]]:
    """Fake the *constructors* (not the cache) so the real cache sits in between."""
    built: list[tuple[str, str]] = []

    def get_embedder(name="fake-e", device="auto"):
        built.append(("embedder", device))
        return _fake_embedder(name, device)

    def get_reranker(name="fake-r", device="auto"):
        built.append(("reranker", device))
        return (lambda q, docs: [float(len(d)) for d in docs]), name

    monkeypatch.setattr(mc, "get_embedder", get_embedder)
    monkeypatch.setattr(mc, "get_reranker", get_reranker)
    monkeypatch.setattr(mc, "resolve_device", lambda d="auto": "cpu" if d in ("", "auto") else d)
    monkeypatch.setattr(mc, "_release_memory", lambda: None)
    mc.rag_model_cache.release_all()
    return built


@pytest.fixture
def corpus(tmp_path: Path, constructed: list[tuple[str, str]]) -> PortableRAG:
    src = tmp_path / "src"
    src.mkdir()
    (src / "a.txt").write_text("Descale the kettle every four weeks with citric acid.", encoding="utf-8")
    (src / "b.txt").write_text("The mascot is a blue heron named Pip.", encoding="utf-8")
    rag = PortableRAG(tmp_path / "corpus")
    rag.build_from_directory(src, name="cache-test", embedder="fake-e", device="cpu")
    return rag


def test_repeated_load_and_search_build_models_once(corpus: PortableRAG, constructed) -> None:
    constructed.clear()
    mc.rag_model_cache.release_all()
    for _ in range(4):
        q = corpus.load()
        hits = q.search("how often to descale", top_k=2, rerank=True)
        assert hits
    assert constructed.count(("embedder", "cpu")) == 1


def test_reranker_is_cached_across_queries(corpus: PortableRAG, constructed) -> None:
    constructed.clear()
    for _ in range(3):
        q = corpus.load()  # a fresh PortableRAGQuery per request, like the routes
        q.manifest.rag_settings.rerank_enabled = True
        q.manifest.rag_settings.reranker = "fake-r"
        assert q.search("mascot", top_k=1, rerank=True)
    assert constructed.count(("reranker", "cpu")) == 1


def test_release_forces_a_rebuild_on_next_load(corpus: PortableRAG, constructed) -> None:
    corpus.load()
    constructed.clear()
    assert mc.release_rag_models("test") >= 1
    corpus.load()
    assert ("embedder", "cpu") in constructed


# ── release points ──


@pytest.fixture
def released(monkeypatch: pytest.MonkeyPatch) -> list[str]:
    reasons: list[str] = []
    monkeypatch.setattr(mc, "release_rag_models", lambda reason="": reasons.append(reason) or 0)
    return reasons


def test_training_start_releases_rag_models(released: list[str]) -> None:
    from finetune_studio.training.engine import TrainingConfig, TrainingEngine

    def worker(*args, **kwargs):
        args[-1].put({"op": "done"}) if hasattr(args[-1], "put") else None

    eng = TrainingEngine()
    try:
        eng.start(TrainingConfig(model_path="m", output_dir="out", unsloth=False),
                  [{"messages": []}], "", _worker_target=worker)
    finally:
        eng.stop()
    assert released == ["training start"]


def test_inference_engine_load_releases_rag_models(released: list[str], monkeypatch) -> None:
    from finetune_studio.testing.inference import InferenceEngine

    eng = InferenceEngine()
    monkeypatch.setattr(eng, "_load_hf", lambda *a, **k: None)
    monkeypatch.setattr(eng, "_start_idle_timer", lambda: None)
    eng.load("/nonexistent/model-dir")
    assert released == ["model load"]


def test_unload_endpoints_release_rag_models(client, released: list[str], monkeypatch) -> None:
    monkeypatch.setattr("finetune_studio.models.llama_loader.unload_all_models", lambda: None)
    assert client.post("/api/models/unload").status_code == 200
    assert client.post("/api/providers/unload").status_code == 200
    assert released == ["unload all models", "unload all models"]


def test_status_reports_rag_cache(client) -> None:
    from finetune_studio.webui.app import inference_engine
    inference_engine.model = None  # the client fixture mocks the engine
    body = client.get("/api/inference/status").json()
    assert body["rag_models"]["loaded"] == 0
    assert "idle_timeout" in body["rag_models"]
