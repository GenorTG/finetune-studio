"""Regression: HF hub cache snapshots get their repo name, not a generic arch label.

Layout is models--<org>--<repo>/snapshots/<40-hex>. The old code took the hash
dir's parent ("snapshots"), so every cached Qwen3 showed as "Qwen3 (qwen3)" in
the base-model dropdown and couldn't be told apart.
"""
from __future__ import annotations

from finetune_studio.models.registry import _safe_model_name

_HASH = "e86f5289a2ca40ab7114071959dc2079cf97cb3c"
_QWEN_CFG = {"model_type": "qwen3", "architectures": ["Qwen3ForCausalLM"]}


def test_hf_snapshot_uses_repo_name() -> None:
    root = f"/home/u/.cache/huggingface/hub/models--unsloth--qwen3-0.6b-unsloth-bnb-4bit/snapshots/{_HASH}"
    assert _safe_model_name(root, _QWEN_CFG) == "qwen3-0.6b-unsloth-bnb-4bit"


def test_distinct_snapshots_get_distinct_names() -> None:
    a = _safe_model_name(f"/c/hub/models--Qwen--Qwen3-0.6B/snapshots/{_HASH}", _QWEN_CFG)
    b = _safe_model_name(f"/c/hub/models--Qwen--Qwen3-4B/snapshots/{_HASH}", _QWEN_CFG)
    assert a == "Qwen3-0.6B"
    assert b == "Qwen3-4B"


def test_plain_hash_dir_without_snapshots_still_uses_parent() -> None:
    assert _safe_model_name(f"/models/my-model/{_HASH}", _QWEN_CFG) == "my-model"
