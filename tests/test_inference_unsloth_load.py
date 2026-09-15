"""Regression tests for Unsloth HF load path (QABUG-014-runtime)."""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any
from unittest.mock import MagicMock

import pytest

from finetune_studio.testing.inference import InferenceEngine


def test_looks_like_qwen3_from_path() -> None:
    assert InferenceEngine._looks_like_qwen3("/out/qwen3-0.6b/merged") is True
    assert InferenceEngine._looks_like_qwen3("/models/llama-3-8b") is False


def test_looks_like_qwen3_from_config(tmp_path: Path) -> None:
    merged = tmp_path / "merged"
    merged.mkdir()
    (merged / "config.json").write_text(
        json.dumps({"model_type": "qwen3", "architectures": ["Qwen3ForCausalLM"]}),
        encoding="utf-8",
    )
    assert InferenceEngine._looks_like_qwen3(str(merged)) is True


def test_load_hf_prefers_fastqwen3_when_available(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Qwen3 checkpoints must go through FastQwen3Model.from_pretrained."""
    import sys
    import types

    merged = tmp_path / "qwen3-merged"
    merged.mkdir()
    (merged / "config.json").write_text(
        json.dumps({"model_type": "qwen3"}),
        encoding="utf-8",
    )

    calls: dict[str, Any] = {}
    fake_model = MagicMock(name="unsloth_model")
    fake_tok = MagicMock(name="tokenizer")
    fake_tok.pad_token = None
    fake_tok.eos_token = "</s>"

    class FakeFastQwen3:
        @staticmethod
        def from_pretrained(**kwargs: Any) -> tuple[Any, Any]:
            calls["kwargs"] = kwargs
            calls["loader"] = "FastQwen3Model"
            return fake_model, fake_tok

        @staticmethod
        def for_inference(model: Any) -> None:
            calls["for_inference"] = model

    class FakeFastLanguage:
        @staticmethod
        def from_pretrained(**kwargs: Any) -> tuple[Any, Any]:
            calls["kwargs"] = kwargs
            calls["loader"] = "FastLanguageModel"
            return fake_model, fake_tok

        @staticmethod
        def for_inference(model: Any) -> None:
            calls["for_inference"] = model

    unsloth_mod = types.ModuleType("unsloth")
    unsloth_mod.FastLanguageModel = FakeFastLanguage  # type: ignore[attr-defined]
    models_mod = types.ModuleType("unsloth.models")
    qwen3_mod = types.ModuleType("unsloth.models.qwen3")
    qwen3_mod.FastQwen3Model = FakeFastQwen3  # type: ignore[attr-defined]
    unsloth_mod.models = models_mod  # type: ignore[attr-defined]

    monkeypatch.setitem(sys.modules, "unsloth", unsloth_mod)
    monkeypatch.setitem(sys.modules, "unsloth.models", models_mod)
    monkeypatch.setitem(sys.modules, "unsloth.models.qwen3", qwen3_mod)

    engine = InferenceEngine()
    engine._load_hf(str(merged), device="auto", max_seq_length=2048, load_in_4bit=True)

    assert calls["loader"] == "FastQwen3Model"
    assert calls["kwargs"]["model_name"] == str(merged)
    assert calls["kwargs"]["max_seq_length"] == 2048
    assert calls["kwargs"]["load_in_4bit"] is True
    assert calls["for_inference"] is fake_model
    assert engine.model is fake_model
    assert engine.tokenizer is fake_tok
    assert fake_tok.pad_token == fake_tok.eos_token
    assert engine.is_gguf is False


def test_load_hf_falls_back_without_unsloth(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """When Unsloth is missing, vanilla transformers path still runs."""
    merged = tmp_path / "plain"
    merged.mkdir()
    (merged / "config.json").write_text("{}", encoding="utf-8")

    import sys

    fake_model = MagicMock()
    fake_model.device = "cpu"
    fake_tok = MagicMock()
    fake_tok.pad_token = "<pad>"
    fake_tok.eos_token = "</s>"

    class FakeAutoTok:
        @staticmethod
        def from_pretrained(*_a: Any, **_k: Any) -> Any:
            return fake_tok

    class FakeAutoModel:
        @staticmethod
        def from_pretrained(*_a: Any, **_k: Any) -> Any:
            return fake_model

    fake_tf = SimpleNamespace(
        AutoTokenizer=FakeAutoTok,
        AutoModelForCausalLM=FakeAutoModel,
    )
    monkeypatch.setitem(sys.modules, "transformers", fake_tf)

    engine = InferenceEngine()
    monkeypatch.setattr(engine, "_load_hf_unsloth", lambda *_a, **_k: False)
    engine._load_hf(str(merged), device="auto")
    assert engine.model is fake_model
    assert engine.tokenizer is fake_tok
