"""Tests for the model_label Jinja filter / helper (E2E-42)."""

from __future__ import annotations

from finetune_studio.webui.model_labels import model_label
from finetune_studio.webui.routes.pages import templates


def test_hf_cache_snapshot_path_to_org_repo() -> None:
    path = (
        "/home/user/.cache/huggingface/hub/"
        "models--Qwen--Qwen3-0.6B/snapshots/e86f5289a2caabcdef0123456789"
    )
    assert model_label(path) == "Qwen/Qwen3-0.6B"


def test_hf_cache_with_nested_repo_name() -> None:
    path = (
        "/data/hf/hub/models--meta-llama--Llama-3.2-1B-Instruct/"
        "snapshots/abc123"
    )
    assert model_label(path) == "meta-llama/Llama-3.2-1B-Instruct"


def test_plain_hf_id_kept() -> None:
    assert model_label("Qwen/Qwen3-0.6B") == "Qwen/Qwen3-0.6B"


def test_local_dir_uses_last_segment() -> None:
    assert model_label("/projects/foo/output/run-1/merged") == "merged"


def test_empty_and_none() -> None:
    assert model_label(None) == ""
    assert model_label("") == ""
    assert model_label("   ") == ""


def test_filter_registered_on_pages_templates() -> None:
    assert "model_label" in templates.env.filters
    assert templates.env.filters["model_label"]("org/repo") == "org/repo"
