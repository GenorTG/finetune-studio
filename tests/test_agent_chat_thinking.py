"""E2E-18: Qwen3 thinking-tag stripping + truncated tool-call detection."""

from __future__ import annotations

from finetune_studio.webui.routes.data_prep_chat import (
    _TRUNCATION_MSG,
    _extract_tool_calls,
    _looks_truncated,
    _strip_thinking_reply,
)


def test_strip_bare_think_closing_tag() -> None:
    """Qwen3 chat template injects opening <think>; reply has only </think>."""
    raw = (
        "I should list sources first.\n"
        "</think>\n"
        "I'll call list_sources now."
    )
    assert _strip_thinking_reply(raw) == "I'll call list_sources now."


def test_strip_paired_think_still_works() -> None:
    raw = "<think>secret reasoning</think>\nVisible answer."
    assert _strip_thinking_reply(raw) == "Visible answer."


def test_tool_call_after_bare_think_is_parsed() -> None:
    raw = (
        "Planning to create pairs…\n"
        "</think>\n"
        '<tool_call>{"name":"create_qa_pairs","arguments":'
        '{"source_id":"abcd1234","pairs":[{"question":"Q?","answer":"A."}]}}'
        "</tool_call>"
    )
    calls = _extract_tool_calls(raw)
    assert len(calls) == 1
    assert calls[0]["name"] == "create_qa_pairs"
    assert calls[0]["arguments"]["source_id"] == "abcd1234"
    assert not _looks_truncated(raw)


def test_truncated_tool_call_produces_cutoff_message() -> None:
    """Incomplete <tool_call> JSON → truncation detector + user-facing msg."""
    raw = (
        "Reasoning about the file…\n"
        "</think>\n"
        '<tool_call>{"name":"create_qa_pairs","arguments":{"source_id'
    )
    assert _extract_tool_calls(raw) == []
    assert _looks_truncated(raw) is True
    msg = _TRUNCATION_MSG.format(n=1024)
    assert "max_tokens=1024" in msg
    assert "raise Max tokens and retry" in msg


def test_unclosed_think_block_looks_truncated() -> None:
    raw = "<think>still thinking with no closer and no tool call"
    assert _looks_truncated(raw) is True
