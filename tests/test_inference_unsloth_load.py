"""E2E-40: HF inference uses plain transformers (never Unsloth)."""

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


def test_load_hf_uses_transformers_not_unsloth(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    merged = tmp_path / "qwen3-merged"
    merged.mkdir()
    (merged / "config.json").write_text(
        json.dumps({"model_type": "qwen3"}),
        encoding="utf-8",
    )

    import sys

    fake_model = MagicMock(name="hf_model")
    fake_model.device = "cpu"
    fake_tok = MagicMock(name="tokenizer")
    fake_tok.pad_token = None
    fake_tok.eos_token = "</s>"
    calls: dict[str, Any] = {}

    class FakeAutoTok:
        @staticmethod
        def from_pretrained(*_a: Any, **_k: Any) -> Any:
            return fake_tok

    class FakeAutoModel:
        @staticmethod
        def from_pretrained(*_a: Any, **kwargs: Any) -> Any:
            calls["kwargs"] = kwargs
            return fake_model

    fake_tf = SimpleNamespace(
        AutoTokenizer=FakeAutoTok,
        AutoModelForCausalLM=FakeAutoModel,
        BitsAndBytesConfig=MagicMock(),
    )
    monkeypatch.setitem(sys.modules, "transformers", fake_tf)
    monkeypatch.setattr(
        "torch.cuda.is_available", lambda: False, raising=False,
    )

    # If Unsloth sneaks in, fail loudly.
    monkeypatch.setitem(sys.modules, "unsloth", MagicMock())

    engine = InferenceEngine()
    engine._load_hf(str(merged), device="auto", max_seq_length=2048, load_in_4bit=False)

    assert engine.model is fake_model
    assert engine.tokenizer is fake_tok
    assert fake_tok.pad_token == fake_tok.eos_token
    assert "quantization_config" not in calls.get("kwargs", {})
    assert "unsloth" not in str(type(engine.model)).lower()


def test_load_hf_4bit_uses_bitsandbytes_config(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    merged = tmp_path / "plain"
    merged.mkdir()
    (merged / "config.json").write_text("{}", encoding="utf-8")

    import sys

    fake_model = MagicMock()
    fake_tok = MagicMock()
    fake_tok.pad_token = "<pad>"
    bnb_calls: list[dict[str, Any]] = []

    class FakeBnb:
        def __init__(self, **kwargs: Any) -> None:
            bnb_calls.append(kwargs)

    class FakeAutoTok:
        @staticmethod
        def from_pretrained(*_a: Any, **_k: Any) -> Any:
            return fake_tok

    class FakeAutoModel:
        @staticmethod
        def from_pretrained(*_a: Any, **kwargs: Any) -> Any:
            assert "quantization_config" in kwargs
            return fake_model

    monkeypatch.setitem(
        sys.modules,
        "transformers",
        SimpleNamespace(
            AutoTokenizer=FakeAutoTok,
            AutoModelForCausalLM=FakeAutoModel,
            BitsAndBytesConfig=FakeBnb,
        ),
    )
    monkeypatch.setattr("torch.cuda.is_available", lambda: True)

    engine = InferenceEngine()
    engine._load_hf(str(merged), device="auto", load_in_4bit=True)
    assert engine.model is fake_model
    assert bnb_calls and bnb_calls[0].get("load_in_4bit") is True


def test_generate_hf_strips_bare_think_closer() -> None:
    engine = InferenceEngine()
    engine.model = MagicMock()
    engine.model.device = "cpu"

    import torch

    class _Batch(dict):
        def to(self, _device: Any) -> "_Batch":
            return self

    tok = MagicMock()
    tok.apply_chat_template.return_value = "PROMPT"
    tok.pad_token_id = 0
    tok.return_value = _Batch(input_ids=torch.tensor([[1, 2, 3]]))
    tok.decode.return_value = "</think>\n\nParis."
    engine.tokenizer = tok
    engine.model.generate.return_value = torch.tensor([[1, 2, 3, 4, 5]])

    out = engine._generate_hf(
        [{"role": "user", "content": "capital?"}],
        max_tokens=8,
        temperature=0.0,
        top_p=0.9,
        top_k=40,
        repeat_penalty=1.0,
        stop=None,
        think=False,
    )
    assert out == "Paris."
    assert "</think>" not in out
