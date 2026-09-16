"""Regression: incomplete shared ST dirs → Pooling missing embedding_dimension."""
from __future__ import annotations

import json
import shutil
from pathlib import Path
from unittest.mock import patch

import pytest

from finetune_studio.data import sentence_transformer_local as stl
from finetune_studio.data import shared_models as sm
from finetune_studio.data.rag_portable.constants import EMBEDDER_LOCAL_PREFIX
from finetune_studio.data.sentence_transformer_local import (
    copy_sentence_transformer_tree,
    is_complete_sentence_transformer_dir,
    prepare_local_sentence_transformer_dir,
)


def _write_mini_st_dir(root: Path, *, with_pooling: bool = True, dim: int = 384) -> Path:
    """Minimal on-disk SentenceTransformer layout (no real weights needed)."""
    root.mkdir(parents=True, exist_ok=True)
    (root / "config.json").write_text(
        json.dumps({"hidden_size": dim, "model_type": "bert"}),
        encoding="utf-8",
    )
    (root / "modules.json").write_text(
        json.dumps(
            [
                {
                    "idx": 0,
                    "name": "0",
                    "path": "",
                    "type": "sentence_transformers.base.modules.transformer.Transformer",
                },
                {
                    "idx": 1,
                    "name": "1",
                    "path": "1_Pooling",
                    "type": (
                        "sentence_transformers.sentence_transformer"
                        ".modules.pooling.Pooling"
                    ),
                },
                {
                    "idx": 2,
                    "name": "2",
                    "path": "2_Normalize",
                    "type": (
                        "sentence_transformers.sentence_transformer"
                        ".modules.normalize.Normalize"
                    ),
                },
            ]
        ),
        encoding="utf-8",
    )
    (root / "model.safetensors").write_bytes(b"fake-weights")
    if with_pooling:
        (root / "1_Pooling").mkdir()
        (root / "1_Pooling" / "config.json").write_text(
            json.dumps(
                {
                    "embedding_dimension": dim,
                    "pooling_mode": "mean",
                    "include_prompt": True,
                }
            ),
            encoding="utf-8",
        )
        (root / "2_Normalize").mkdir()
    return root


def test_incomplete_dir_is_not_complete(tmp_path: Path) -> None:
    broken = _write_mini_st_dir(tmp_path / "broken", with_pooling=False)
    assert not is_complete_sentence_transformer_dir(broken)
    complete = _write_mini_st_dir(tmp_path / "ok", with_pooling=True)
    assert is_complete_sentence_transformer_dir(complete)


def test_prepare_repairs_missing_pooling_config(tmp_path: Path) -> None:
    """Mirrors production: modules.json present, 1_Pooling/ dropped by flat copy."""
    broken = _write_mini_st_dir(tmp_path / "broken", with_pooling=False, dim=384)
    assert not (broken / "1_Pooling" / "config.json").exists()

    prepare_local_sentence_transformer_dir(broken)

    cfg = json.loads((broken / "1_Pooling" / "config.json").read_text(encoding="utf-8"))
    assert cfg["embedding_dimension"] == 384
    assert cfg["pooling_mode"] == "mean"
    assert (broken / "2_Normalize").is_dir()
    assert is_complete_sentence_transformer_dir(broken)


def test_prepare_normalizes_legacy_word_embedding_dimension(tmp_path: Path) -> None:
    root = _write_mini_st_dir(tmp_path / "legacy", with_pooling=True, dim=384)
    (root / "1_Pooling" / "config.json").write_text(
        json.dumps(
            {
                "word_embedding_dimension": 384,
                "pooling_mode_mean_tokens": True,
            }
        ),
        encoding="utf-8",
    )
    prepare_local_sentence_transformer_dir(root)
    cfg = json.loads((root / "1_Pooling" / "config.json").read_text(encoding="utf-8"))
    assert cfg["embedding_dimension"] == 384
    assert "pooling_mode" in cfg


def test_prepare_raises_when_dimension_uninferable(tmp_path: Path) -> None:
    broken = _write_mini_st_dir(tmp_path / "nodim", with_pooling=False)
    (broken / "config.json").write_text("{}", encoding="utf-8")
    with pytest.raises(RuntimeError, match="embedding_dimension"):
        prepare_local_sentence_transformer_dir(broken)


def test_copy_preserves_pooling_subdir(tmp_path: Path) -> None:
    src = _write_mini_st_dir(tmp_path / "src", with_pooling=True)
    dest = tmp_path / "dest"
    copy_sentence_transformer_tree(src, dest)
    assert (dest / "1_Pooling" / "config.json").is_file()
    assert (dest / "2_Normalize").is_dir()
    assert is_complete_sentence_transformer_dir(dest)


def test_register_from_src_preserves_module_subdirs(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Shared-store register must not flatten 1_Pooling into the root."""
    emb_root = tmp_path / "embedders"
    monkeypatch.setattr(sm, "EMBEDDERS", emb_root)
    monkeypatch.setattr(sm, "SHARED", tmp_path)
    monkeypatch.setattr(sm, "RERANKERS", tmp_path / "rerankers")

    src = _write_mini_st_dir(tmp_path / "src_model", with_pooling=True, dim=384)
    ref = sm.register("all-MiniLM-L6-v2", kind="embedder", src_dir=src, prefer_cached=False)
    target = Path(ref.path)
    assert (target / "1_Pooling" / "config.json").is_file()
    assert (target / "2_Normalize").is_dir()
    # Must not have overwritten root config with pooling config
    root_cfg = json.loads((target / "config.json").read_text(encoding="utf-8"))
    assert root_cfg.get("hidden_size") == 384
    assert is_complete_sentence_transformer_dir(target)


def test_register_replaces_incomplete_cached_embedder(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    emb_root = tmp_path / "embedders"
    monkeypatch.setattr(sm, "EMBEDDERS", emb_root)
    monkeypatch.setattr(sm, "SHARED", tmp_path)
    monkeypatch.setattr(sm, "RERANKERS", tmp_path / "rerankers")

    broken_src = _write_mini_st_dir(tmp_path / "broken_src", with_pooling=False)
    # First register with a flat tree (simulates old buggy cache)
    h = sm.content_hash(broken_src)
    target = emb_root / f"all-MiniLM-L6-v2@{h}"
    target.mkdir(parents=True)
    for f in broken_src.iterdir():
        if f.is_file():
            shutil.copy2(f, target / f.name)
    assert not is_complete_sentence_transformer_dir(target)

    good_src = _write_mini_st_dir(tmp_path / "good_src", with_pooling=True)
    # Same safetensors → same content_hash key
    (good_src / "model.safetensors").write_bytes(b"fake-weights")
    ref = sm.register(
        "all-MiniLM-L6-v2", kind="embedder", src_dir=good_src, prefer_cached=True
    )
    assert Path(ref.path) == target
    assert is_complete_sentence_transformer_dir(Path(ref.path))


def test_load_sentence_transformer_prepares_before_construct(
    tmp_path: Path,
) -> None:
    """get_embedder local path must prepare dir before SentenceTransformer()."""
    broken = _write_mini_st_dir(tmp_path / "local", with_pooling=False, dim=384)
    prepared: list[Path] = []
    real_prepare = stl.prepare_local_sentence_transformer_dir

    class _FakeST:
        def __init__(
            self, path: str, device: str = "cpu", cache_folder: str | None = None
        ):
            p = Path(path)
            if not (p / "1_Pooling" / "config.json").is_file():
                raise TypeError(
                    "Pooling.__init__() missing 1 required positional argument: "
                    "'embedding_dimension'"
                )
            self._dim = 384

        def get_embedding_dimension(self) -> int:
            return self._dim

        def encode(self, texts, **kwargs):  # type: ignore[no-untyped-def]
            import numpy as np

            n = 1 if isinstance(texts, str) else len(list(texts))
            return np.zeros((n, 384), dtype=np.float32)

    def _track_prepare(path: Path) -> Path:
        prepared.append(Path(path))
        return real_prepare(path)

    with (
        patch.object(stl, "prepare_local_sentence_transformer_dir", side_effect=_track_prepare),
        patch("sentence_transformers.SentenceTransformer", _FakeST),
    ):
        from finetune_studio.data.rag_portable.embedders import get_embedder

        encode, info = get_embedder(f"{EMBEDDER_LOCAL_PREFIX}{broken}", device="cpu")
        assert info.dim == 384
        assert prepared == [broken]
        assert (broken / "1_Pooling" / "config.json").is_file()
        vec = encode("hello")
        assert vec.shape == (384,)


def test_get_embedder_does_not_fallback_on_prepare_failure(
    tmp_path: Path,
) -> None:
    """Unrepairable local dir must raise — never DEFAULT_EMBEDDER."""
    broken = _write_mini_st_dir(tmp_path / "bad", with_pooling=False)
    (broken / "config.json").write_text("{}", encoding="utf-8")

    with pytest.raises(RuntimeError, match="embedding_dimension"):
        from finetune_studio.data.rag_portable.embedders import get_embedder

        get_embedder(f"{EMBEDDER_LOCAL_PREFIX}{broken}", device="cpu")
