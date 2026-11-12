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
    root = f"/srv/u/.cache/huggingface/hub/models--unsloth--qwen3-0.6b-unsloth-bnb-4bit/snapshots/{_HASH}"
    assert _safe_model_name(root, _QWEN_CFG) == "qwen3-0.6b-unsloth-bnb-4bit"


def test_distinct_snapshots_get_distinct_names() -> None:
    a = _safe_model_name(f"/c/hub/models--Qwen--Qwen3-0.6B/snapshots/{_HASH}", _QWEN_CFG)
    b = _safe_model_name(f"/c/hub/models--Qwen--Qwen3-4B/snapshots/{_HASH}", _QWEN_CFG)
    assert a == "Qwen3-0.6B"
    assert b == "Qwen3-4B"


def test_plain_hash_dir_without_snapshots_still_uses_parent() -> None:
    assert _safe_model_name(f"/models/my-model/{_HASH}", _QWEN_CFG) == "my-model"


def test_repeated_export_names_get_their_run_id() -> None:
    """Two unpinned runs of one project and base both read `<project> · <base> · merged`: tag them with the run."""
    from finetune_studio.models.registry import ModelInfo, _disambiguate_run_names

    name = "quill · Llama-3.2-3B-Instruct · merged"
    models = _disambiguate_run_names([
        ModelInfo(name=name, path="output/projects/p/runs/fbf57aa1/merged", format="safetensors", size_gb=6.0,
                  category="trained_export", project_id="p", run_id="fbf57aa1c0ffee"),
        ModelInfo(name=name, path="output/projects/p/runs/a8efac7e/merged", format="safetensors", size_gb=6.0,
                  category="trained_export", project_id="p", run_id="a8efac7e"),
        ModelInfo(name="quill · Qwen3.5-9B · GGUF · Q4_K_M", path="output/projects/p/runs/7953987f/gguf/m.gguf",
                  format="gguf", size_gb=5.2, category="trained_export", project_id="p", run_id="7953987f"),
    ])
    assert [m.name for m in models] == [
        f"{name} · run fbf57aa1",
        f"{name} · run a8efac7e",
        "quill · Qwen3.5-9B · GGUF · Q4_K_M",  # unique: no hash
    ]
