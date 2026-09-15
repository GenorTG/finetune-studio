"""Focused tests for GPTQ detection and gptqmodel Torch-backend load."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any
from unittest.mock import MagicMock

import pytest

from finetune_studio.testing import gptq_load
from finetune_studio.testing.inference import InferenceEngine


def test_detects_quantize_config_json(tmp_path: Path) -> None:
    gptq = tmp_path / "gptq"
    gptq.mkdir()
    (gptq / "config.json").write_text("{}", encoding="utf-8")
    (gptq / "quantize_config.json").write_text(
        json.dumps({"quant_method": "gptq", "bits": 4}),
        encoding="utf-8",
    )
    assert gptq_load.is_local_gptq_checkpoint(str(gptq)) is True


def test_detects_quantization_config_metadata(tmp_path: Path) -> None:
    gptq = tmp_path / "gptq-meta"
    gptq.mkdir()
    (gptq / "config.json").write_text(
        json.dumps(
            {
                "model_type": "qwen3",
                "quantization_config": {
                    "quant_method": "gptq",
                    "format": "gptq",
                    "bits": 4,
                },
            }
        ),
        encoding="utf-8",
    )
    assert gptq_load.is_local_gptq_checkpoint(str(gptq)) is True


def test_plain_hf_dir_is_not_gptq(tmp_path: Path) -> None:
    plain = tmp_path / "merged"
    plain.mkdir()
    (plain / "config.json").write_text(
        json.dumps({"model_type": "qwen3"}),
        encoding="utf-8",
    )
    assert gptq_load.is_local_gptq_checkpoint(str(plain)) is False


def test_file_path_is_not_gptq(tmp_path: Path) -> None:
    gguf = tmp_path / "model.gguf"
    gguf.write_bytes(b"GGUF")
    assert gptq_load.is_local_gptq_checkpoint(str(gguf)) is False


def test_load_gptq_uses_from_quantized_gptq_torch(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    gptq = tmp_path / "gptq"
    gptq.mkdir()
    (gptq / "config.json").write_text("{}", encoding="utf-8")
    (gptq / "quantize_config.json").write_text("{}", encoding="utf-8")

    calls: dict[str, Any] = {}
    fake_model = MagicMock(name="gptq_model")
    fake_backend = SimpleNamespace(GPTQ_TORCH="gptq_torch")

    class _FakeGPTQModel:
        @staticmethod
        def from_quantized(path: str, **kwargs: Any) -> Any:
            calls["path"] = path
            calls["kwargs"] = kwargs
            return fake_model

    monkeypatch.setitem(
        sys.modules,
        "gptqmodel",
        SimpleNamespace(GPTQModel=_FakeGPTQModel, BACKEND=fake_backend),
    )

    loaded = gptq_load.load_gptq_model_torch(str(gptq), device_map={"": 0})
    assert loaded is fake_model
    assert calls["path"] == str(gptq)
    assert calls["kwargs"]["backend"] == "gptq_torch"
    assert calls["kwargs"]["trust_remote_code"] is True
    assert calls["kwargs"]["device_map"] == {"": 0}


def test_load_gptq_missing_package_raises_actionable(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    real_import = __import__

    def _block_gptqmodel(name: str, *args: Any, **kwargs: Any) -> Any:
        if name == "gptqmodel" or name.startswith("gptqmodel."):
            raise ImportError("no gptqmodel")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr("builtins.__import__", _block_gptqmodel)
    with pytest.raises(RuntimeError, match=r"gptqmodel.*BACKEND\.GPTQ_TORCH"):
        gptq_load.load_gptq_model_torch("/tmp/unused")


def test_load_gptq_missing_backend_raises_actionable(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setitem(
        sys.modules,
        "gptqmodel",
        SimpleNamespace(
            GPTQModel=MagicMock(),
            BACKEND=SimpleNamespace(),  # no GPTQ_TORCH
        ),
    )
    with pytest.raises(RuntimeError, match=r"BACKEND\.GPTQ_TORCH"):
        gptq_load.load_gptq_model_torch("/tmp/unused")


def test_inference_load_hf_routes_gptq_to_torch_backend(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    gptq = tmp_path / "run-gptq"
    gptq.mkdir()
    (gptq / "config.json").write_text(
        json.dumps({"quantization_config": {"quant_method": "gptq"}}),
        encoding="utf-8",
    )
    (gptq / "quantize_config.json").write_text("{}", encoding="utf-8")

    fake_model = MagicMock(name="gptq_model")
    fake_tok = MagicMock(name="tokenizer")
    fake_tok.pad_token = None
    fake_tok.eos_token = "</s>"
    auto_model_calls: list[Any] = []

    class FakeAutoTok:
        @staticmethod
        def from_pretrained(*_a: Any, **_k: Any) -> Any:
            return fake_tok

    class FakeAutoModel:
        @staticmethod
        def from_pretrained(*_a: Any, **_k: Any) -> Any:
            auto_model_calls.append((_a, _k))
            raise AssertionError("plain HF load must not run for GPTQ dirs")

    monkeypatch.setitem(
        sys.modules,
        "transformers",
        SimpleNamespace(
            AutoTokenizer=FakeAutoTok,
            AutoModelForCausalLM=FakeAutoModel,
            BitsAndBytesConfig=MagicMock(),
        ),
    )
    monkeypatch.setattr("torch.cuda.is_available", lambda: False, raising=False)

    load_calls: dict[str, Any] = {}

    def _fake_load(path: str, *, device_map: Any = None) -> Any:
        load_calls["path"] = path
        load_calls["device_map"] = device_map
        return fake_model

    monkeypatch.setattr(
        "finetune_studio.testing.gptq_load.load_gptq_model_torch",
        _fake_load,
    )

    engine = InferenceEngine()
    engine._load_hf(str(gptq), device="auto", load_in_4bit=False)

    assert engine.model is fake_model
    assert engine.tokenizer is fake_tok
    assert fake_tok.pad_token == fake_tok.eos_token
    assert load_calls["path"] == str(gptq)
    assert load_calls["device_map"] == "cpu"
    assert auto_model_calls == []
    assert engine.is_gguf is False
