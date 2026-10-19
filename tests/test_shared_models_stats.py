"""Shared-models stats API + RAG page contract for embedders/rerankers."""
from __future__ import annotations

import json
from pathlib import Path

from finetune_studio.data import shared_models as sm


def test_human_size_formats() -> None:
    assert sm.human_size(500) == "500 B"
    assert sm.human_size(2048).endswith("KB")
    assert "MB" in sm.human_size(3 * 1024 * 1024)


def test_stats_flattens_embedders_and_rerankers(
    tmp_path: Path, monkeypatch
) -> None:
    root = tmp_path / "shared_models"
    emb = root / "embedders" / "intfloat__multilingual-e5-large@abc12345"
    rer = root / "rerankers" / "BAAI__bge-reranker-base@def67890"
    emb.mkdir(parents=True)
    rer.mkdir(parents=True)
    (emb / "weights.bin").write_bytes(b"e" * 1500)
    (rer / "weights.bin").write_bytes(b"r" * 2500)
    (emb / "META.json").write_text(
        json.dumps({
            "name": "intfloat/multilingual-e5-large",
            "kind": "embedder",
            "short_id": emb.name,
            "use_count": 2,
        }),
        encoding="utf-8",
    )
    (rer / "META.json").write_text(
        json.dumps({
            "name": "BAAI/bge-reranker-base",
            "kind": "reranker",
            "short_id": rer.name,
            "use_count": 1,
        }),
        encoding="utf-8",
    )

    monkeypatch.setattr(sm, "SHARED", root)
    monkeypatch.setattr(sm, "EMBEDDERS", root / "embedders")
    monkeypatch.setattr(sm, "RERANKERS", root / "rerankers")

    out = sm.stats()
    assert len(out["embedders"]) == 1
    assert len(out["rerankers"]) == 1
    assert out["total_size_bytes"] > 0
    # Flat list for UIs that only read models/items.
    assert len(out["models"]) == 2
    assert out["items"] == out["models"]
    names = {m["name"] for m in out["models"]}
    assert "intfloat/multilingual-e5-large" in names
    assert "BAAI/bge-reranker-base" in names
    kinds = {m["kind"] for m in out["models"]}
    assert kinds == {"embedder", "reranker"}
    for m in out["models"]:
        assert m["size_human"]
        assert m["size"] == m["size_human"]
        assert m["size_bytes"] > 0


def test_stats_empty_store_is_truthful(tmp_path: Path, monkeypatch) -> None:
    root = tmp_path / "shared_models"
    monkeypatch.setattr(sm, "SHARED", root)
    monkeypatch.setattr(sm, "EMBEDDERS", root / "embedders")
    monkeypatch.setattr(sm, "RERANKERS", root / "rerankers")
    out = sm.stats()
    assert out["embedders"] == []
    assert out["rerankers"] == []
    assert out["models"] == []
    assert out["items"] == []
    assert out["total_size_bytes"] == 0


def test_brain_html_normalizes_embedders_rerankers_shape() -> None:
    html = Path("src/finetune_studio/webui/templates/brain.html").read_text(
        encoding="utf-8",
    )
    assert "normalizeSharedModelList" in html
    assert "d.embedders" in html
    assert "d.rerankers" in html
    assert "No shared models cached yet." in html
    # Must not rely solely on the obsolete flat-only read without fallback.
    assert "humanBytes" in html



def test_register_download_loads_the_model_on_cpu_only(tmp_path: Path, monkeypatch) -> None:
    """Registering only saves the model: it must never take VRAM next to the live embedder."""
    import sys
    import types

    seen: list[dict] = []

    class FakeModel:
        def __init__(self, name: str, *args, **kw) -> None:
            seen.append(kw)

        def save(self, path: str) -> None:
            Path(path, "model.safetensors").write_bytes(b"w")

    monkeypatch.setitem(sys.modules, "sentence_transformers",
                        types.SimpleNamespace(SentenceTransformer=FakeModel, CrossEncoder=FakeModel))
    monkeypatch.setattr(sm, "SHARED", tmp_path / "shared")
    monkeypatch.setattr(sm, "EMBEDDERS", tmp_path / "shared" / "embedders")
    monkeypatch.setattr(sm, "RERANKERS", tmp_path / "shared" / "rerankers")
    monkeypatch.setattr(sm, "hf_cache_dir", lambda: tmp_path / "hf" / "hub")
    real = sm.register.real  # conftest stubs the download branch for every other test
    emb = real("some/embedder", kind="embedder", src_dir=None)
    rer = real("some/reranker", kind="reranker", src_dir=None)
    assert [kw.get("device") for kw in seen] == ["cpu", "cpu"]
    assert Path(emb.path).is_relative_to(tmp_path) and Path(rer.path).is_relative_to(tmp_path)


def test_no_test_reaches_the_shared_model_download() -> None:
    """The conftest stub refuses the download branch like an offline machine, loading nothing."""
    import pytest

    with pytest.raises(OSError, match="tests do not download models"):
        sm.register("intfloat/multilingual-e5-large", kind="embedder")


def test_register_reuses_a_complete_registered_copy(tmp_path: Path, monkeypatch) -> None:
    """Every Smart Brain build re-downloaded and re-saved the 2.2 GB embedder it already had (2026-10-19)."""
    import sys
    import types

    built: list[str] = []

    class FakeModel:
        def __init__(self, name: str, *args, **kw) -> None:
            built.append(name)

        def save(self, path: str) -> None:
            Path(path, "model.safetensors").write_bytes(b"w")
            Path(path, "modules.json").write_text("[]", encoding="utf-8")
            Path(path, "config.json").write_text("{}", encoding="utf-8")

    monkeypatch.setitem(sys.modules, "sentence_transformers",
                        types.SimpleNamespace(SentenceTransformer=FakeModel, CrossEncoder=FakeModel))
    monkeypatch.setattr(sm, "SHARED", tmp_path / "shared")
    monkeypatch.setattr(sm, "EMBEDDERS", tmp_path / "shared" / "embedders")
    monkeypatch.setattr(sm, "RERANKERS", tmp_path / "shared" / "rerankers")
    monkeypatch.setattr(sm, "hf_cache_dir", lambda: tmp_path / "hf" / "hub")
    real = sm.register.real
    first = [real("some/embedder", kind="embedder"), real("some/reranker", kind="reranker")]
    again = [real("some/embedder", kind="embedder"), real("some/reranker", kind="reranker")]
    assert [r.path for r in again] == [r.path for r in first]
    assert built == ["some/embedder", "some/reranker"]          # loaded once each, not per build
    real("some/embedder", kind="embedder", prefer_cached=False)
    assert built[-1] == "some/embedder"                          # an explicit refresh still downloads
