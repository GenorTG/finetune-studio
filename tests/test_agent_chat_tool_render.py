"""Agent chat tool results render readable status; raw JSON stays in Debug."""

from __future__ import annotations

from pathlib import Path

_TEMPLATE = (
    Path(__file__).resolve().parents[1]
    / "src"
    / "finetune_studio"
    / "webui"
    / "templates"
    / "chat_v2.html"
)


def _src() -> str:
    return _TEMPLATE.read_text(encoding="utf-8")


def test_tool_render_shows_no_sources_status() -> None:
    src = _src()
    assert "No sources found" in src
    assert "dp-chat-tool-status" in src
    assert "dp-chat-tool-table" in src


def test_raw_json_is_behind_debug_details() -> None:
    src = _src()
    assert "Debug · raw JSON" in src
    assert "dp-chat-tool-debug" in src
    # Must not force-open a details dump of Full result in the main body.
    assert "det.open = true" not in src
    assert "Full result" not in src.split("function _chatRenderToolCalls", 1)[1].split(
        "async function chatSend", 1
    )[0]


def test_empty_rag_sources_array_renders_status() -> None:
    src = _src()
    body = src.split("function _chatAppend(", 1)[1].split(
        "function _chatRenderToolCalls", 1
    )[0]
    assert "No source chunks retrieved" in body
    assert "Array.isArray(sources)" in body
