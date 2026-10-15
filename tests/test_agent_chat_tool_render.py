"""Guide tool cards render readable status; raw JSON stays behind Debug (the renderer is static/js/guide.js)."""

from __future__ import annotations

from pathlib import Path

_WEBUI = Path(__file__).resolve().parents[1] / "src" / "finetune_studio" / "webui"
_JS = _WEBUI / "static" / "js" / "guide.js"
_TEMPLATE = _WEBUI / "templates" / "chat_v2.html"


def _js() -> str:
    return _JS.read_text(encoding="utf-8")


def _tpl() -> str:
    return _TEMPLATE.read_text(encoding="utf-8")


def test_tool_render_shows_no_sources_status() -> None:
    src = _js()
    assert "No sources found" in src
    assert "guide-tool-status" in src
    assert 'setAttribute("role", "status")' in src


def test_raw_json_is_behind_debug_details() -> None:
    src = _js()
    assert "Debug · raw JSON" in src
    assert "guide-tool-debug" in src
    # Never force-open the raw dump or print a "Full result" block in the card body.
    assert "dbg.open = true" not in src and "Full result" not in src


def test_empty_rag_sources_array_renders_status() -> None:
    """Test mode's source list still reports an empty retrieval explicitly."""
    body = _tpl().split("function _chatAppend(", 1)[1].split("async function chatSend", 1)[0]
    assert "No source chunks retrieved" in body
    assert "Array.isArray(sources)" in body


def test_chat_request_keeps_abort_controller_per_request() -> None:
    src = _tpl()
    body = src.split("async function chatSend()", 1)[1].split("function _chatStop", 1)[0]
    assert "const requestAbort = new AbortController();" in body
    assert "signal: requestAbort.signal" in body
    assert "if (_chatAbort === requestAbort) _chatAbort = null;" in body


def test_agent_mode_hands_off_to_the_docked_guide() -> None:
    """One conversation, one renderer: the chat page no longer has its own agent request path."""
    src = _tpl()
    body = src.split("async function chatSend()", 1)[1].split("function _chatStop", 1)[0]
    assert "window.ftsGuide.ask(text)" in body
    assert "data-prep/chat" not in src
    assert "_chatRenderToolCalls" not in src
