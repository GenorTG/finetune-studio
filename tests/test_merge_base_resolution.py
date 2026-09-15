"""Tests for resolve_merge_base (E2E-32): nf4 → 16-bit sibling lookup."""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from finetune_studio.training.merge_base import (
    MergeBaseNotFound,
    resolve_merge_base,
    strip_quant_suffixes,
)


def _write_config(dir_path: Path, *, quantized: bool) -> None:
    dir_path.mkdir(parents=True, exist_ok=True)
    cfg: dict = {"model_type": "qwen3", "architectures": ["Qwen3ForCausalLM"]}
    if quantized:
        cfg["quantization_config"] = {
            "quant_method": "bitsandbytes",
            "load_in_4bit": True,
            "bnb_4bit_quant_type": "nf4",
        }
    (dir_path / "config.json").write_text(json.dumps(cfg), encoding="utf-8")
    if not quantized:
        # Minimal weight file so the snapshot counts as complete.
        (dir_path / "model.safetensors").write_bytes(b"fake")


def test_strip_quant_suffixes() -> None:
    assert strip_quant_suffixes("qwen3-0.6b-unsloth-bnb-4bit") == "qwen3-0.6b"
    assert strip_quant_suffixes("qwen3-0.6b-bnb-4bit") == "qwen3-0.6b"
    assert strip_quant_suffixes("qwen3-0.6b-4bit") == "qwen3-0.6b"
    assert strip_quant_suffixes("Qwen3-0.6B") == "Qwen3-0.6B"


def test_nf4_unsloth_cache_resolves_to_qwen_sibling(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    home = tmp_path / "home"
    hub = home / ".cache" / "huggingface" / "hub"
    monkeypatch.setenv("HOME", str(home))
    monkeypatch.setenv("HF_HOME", str(home / ".cache" / "huggingface"))
    # Quantized training base (what Unsloth loaded).
    quant_snap = hub / "models--unsloth--qwen3-0.6b-unsloth-bnb-4bit" / "snapshots" / "abc123"
    _write_config(quant_snap, quantized=True)
    # Non-quantized sibling the merge must use.
    fp16_snap = hub / "models--Qwen--Qwen3-0.6B" / "snapshots" / "def456"
    _write_config(fp16_snap, quantized=False)

    got = resolve_merge_base(str(quant_snap))
    assert got == str(fp16_snap)


def test_non_quantized_base_returned_unchanged(tmp_path: Path) -> None:
    base = tmp_path / "Qwen3-0.6B"
    _write_config(base, quantized=False)
    assert resolve_merge_base(str(base)) == str(base)


def test_nothing_local_raises_naming_16bit_repo(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    home = tmp_path / "home"
    monkeypatch.setenv("HOME", str(home))
    monkeypatch.setenv("HF_HOME", str(home / ".cache" / "huggingface"))
    quant = home / "quant-base"
    _write_config(quant, quantized=True)
    # Fake an unsloth-style leaf so bare repo derives to qwen3-0.6b
    named = home / "qwen3-0.6b-unsloth-bnb-4bit"
    _write_config(named, quantized=True)

    with pytest.raises(MergeBaseNotFound) as ei:
        resolve_merge_base(str(named))
    msg = str(ei.value)
    assert "Qwen/" in msg or "qwen3-0.6b" in msg.lower()
    assert "16-bit" in msg.lower() or "Download" in msg
