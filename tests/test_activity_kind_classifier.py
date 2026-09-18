"""Regression test for the operation-kind classifier used by the activity feed.

Pins the routing from API path -> activity-feed kind so future drift is caught.
"""
from __future__ import annotations

import pytest

from finetune_studio.webui.app import _activity_kind

CASES = [
    # Specific kinds (must win over the generic fallbacks).
    ("/api/inference/chat", "inference"),
    ("/api/inference/load", "model_load"),
    ("/api/inference/unload", "model_load"),
    ("/api/chat-v2/load", "model_load"),
    ("/api/chat-v2/unload", "model_load"),
    ("/api/chat-v2/inference/chat", "inference"),
    ("/api/chat-v2/projects/fbcf7083/chat", "inference"),
    ("/api/compare/rag/chat", "inference"),
    # RAG lifecycle and queries.
    ("/api/projects/fbcf7083/rag/build", "rag_build"),
    ("/api/projects/fbcf7083/rag/rebuild", "rag_build"),
    ("/api/projects/fbcf7083/rag/rebuild-vectors", "rag_build"),
    ("/api/projects/fbcf7083/rag/chat", "rag_query"),
    ("/api/projects/fbcf7083/rag/search", "rag_query"),
    # Existing kinds.
    ("/api/projects/fbcf7083/files/upload", "upload"),
    ("/api/projects/fbcf7083/data-prep/upload", "upload"),
    ("/api/projects/fbcf7083/data-prep/sources", "upload"),
    ("/api/projects/fbcf7083/data-prep/chat", "data_prep"),
    ("/api/testing/run-suite", "testing"),
    ("/api/testing/run-rag-suite", "testing"),
    ("/api/benchmarks/projects/fbcf7083/runs/abc/run", "benchmark"),
    ("/api/projects/fbcf7083/merge", "export"),
    ("/api/projects/fbcf7083/export", "export"),
    # Model lifecycle.
    ("/api/models/load", "model_load"),
    ("/api/models/unload", "model_load"),
    ("/api/models/refresh", "model_load"),
    ("/api/providers/openai/test", "model_load"),
    # HF downloads.
    ("/api/hf/download", "download"),
    ("/api/hf/download/cancel", "download"),
    ("/api/hf/local/repo-id", "download"),
    # Training start still wins over operation.
    ("/api/training/start", "training"),
    ("/api/training/runs/abc/start", "training"),
    # Generic operation fallback for anything else.
    ("/api/projects", "operation"),
    ("/api/projects/fbcf7083", "operation"),
    ("/api/projects/fbcf7083/settings", "operation"),
    ("/api/system-update/check", "system_update"),
    ("/api/system/info", "system_update"),
]


@pytest.mark.parametrize("path,expected", CASES)
def test_activity_kind_classification(path: str, expected: str) -> None:
    assert _activity_kind(path) == expected, f"{path} -> {expected}"


def test_rag_query_wins_over_data_prep_chat() -> None:
    """The more specific rag_query kind must win over the broader chat-in-data-prep rule."""
    assert _activity_kind("/api/projects/x/rag/chat") == "rag_query"


def test_inference_wins_over_rag_query_for_chat_v2() -> None:
    """Inference endpoints are classified before rag_query even if 'chat' appears."""
    assert _activity_kind("/api/chat-v2/inference/chat") == "inference"
