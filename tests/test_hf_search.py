"""Regression test for the HF search tokenization + pipeline-tag fallback."""
from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import patch

from finetune_studio.webui.routes.hf_models import SearchRequest, _search_hf


def _m(model_id: str, tags: list[str] | None = None) -> SimpleNamespace:
    return SimpleNamespace(
        modelId=model_id,
        downloads=100,
        likes=10,
        tags=tags or [],
        lastModified="2026-01-01",
        private=False,
    )


def test_free_text_query_matches_repo_id_with_token_split() -> None:
    """A query like 'qwen3 gguf' must match repos containing those tokens."""
    rows = [
        _m("Qwen/Qwen3-8B-GGUF"),
        _m("MaziyarPanahi/Qwen3-14B-GGUF"),
        _m("meta-llama/Llama-3.1-8B-Instruct"),
    ]
    with patch("huggingface_hub.HfApi") as mock_api:
        mock_api.return_value.list_models.return_value = rows
        out = _search_hf(SearchRequest(query="qwen3 gguf", limit=10))
    assert {r["repo_id"] for r in out} == {
        "Qwen/Qwen3-8B-GGUF",
        "MaziyarPanahi/Qwen3-14B-GGUF",
    }


def test_exact_repo_id_still_matches() -> None:
    rows = [_m("Qwen/Qwen3-8B-GGUF"), _m("meta-llama/Llama-3.1-8B-Instruct")]
    with patch("huggingface_hub.HfApi") as mock_api:
        mock_api.return_value.list_models.return_value = rows
        out = _search_hf(SearchRequest(query="Qwen3-8B-GGUF", limit=10))
    assert [r["repo_id"] for r in out] == ["Qwen/Qwen3-8B-GGUF"]


def test_pipeline_tag_filter_relaxes_when_zero_results() -> None:
    """If pipeline_tag='text-generation' yields nothing, retry without it."""
    call_kwargs: list[dict] = []

    def fake_list_models(*_args, **kwargs):
        call_kwargs.append(kwargs)
        if kwargs.get("pipeline_tag") == "text-generation":
            return []  # strict filter returns nothing
        return [_m("Qwen/Qwen3-8B"), _m("Qwen/Qwen3-14B")]

    with patch("huggingface_hub.HfApi") as mock_api:
        mock_api.return_value.list_models.side_effect = fake_list_models
        out = _search_hf(SearchRequest(query="Qwen3", task="text-generation", limit=10))
    assert len(call_kwargs) == 2
    assert call_kwargs[1].get("pipeline_tag") is None
    assert {r["repo_id"] for r in out} == {"Qwen/Qwen3-8B", "Qwen/Qwen3-14B"}


def test_pipeline_tag_strict_filter_kept_when_results_present() -> None:
    rows = [_m("Qwen/Qwen3-8B-GGUF", tags=["text-generation"]),
            _m("Other/NotTextGen", tags=["conversational"])]
    with patch("huggingface_hub.HfApi") as mock_api:
        mock_api.return_value.list_models.return_value = rows
        out = _search_hf(SearchRequest(query="Qwen", task="text-generation", limit=10))
    assert [r["repo_id"] for r in out] == ["Qwen/Qwen3-8B-GGUF"]


def test_empty_query_returns_anything_in_pipeline_tag() -> None:
    rows = [_m("any/repo-A"), _m("any/repo-B"), _m("any/repo-C")]
    with patch("huggingface_hub.HfApi") as mock_api:
        mock_api.return_value.list_models.return_value = rows
        out = _search_hf(SearchRequest(query="", limit=3))
    assert len(out) == 3
