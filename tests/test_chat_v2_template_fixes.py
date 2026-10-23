"""Regression: project chat page fixes found in the human E2E walkthrough.

E2E-13  data-prep's "Open chat in Agent mode" links to ?mode=agent, but the
        page ignored the param and stayed in Test mode.
E2E-14  the inline model picker read m.name / m.size_gb, which /api/hf/local
        doesn't return, rendering "undefined (undefined GB)".
        Native alert() on load failure blocked the page for automation.
"""
from __future__ import annotations

import re
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


def test_mode_query_param_selects_radio() -> None:
    src = _src()
    assert "URLSearchParams(location.search).get('mode')" in src
    assert "dispatchEvent(new Event('change'))" in src


def test_inline_picker_uses_hf_local_fields() -> None:
    body = _src().split("async function chatPopulateInlineLoad(", 1)[1].split("\nfunction ", 1)[0]
    assert "m.repo_id" in body
    assert "m.size_bytes" in body


def test_no_native_alerts() -> None:
    assert re.search(r"(?<![\w.])alert\(", _src()) is None


def test_presets_target_a_real_system_prompt_field() -> None:
    # E2E-15: preset buttons wrote to [name=system_prompt], which didn't exist.
    src = _src()
    assert 'name="system_prompt"' in src
    assert "system_prompt_override:" in src


def test_idle_chat_keeps_empty_state_not_live_success() -> None:
    """Saved localStorage history must not look like a live agent success on idle load."""
    src = _src()
    assert 'id="chat-empty"' in src
    assert "Previous session (saved locally — not a live result)" in src
    assert "_chatRenderPrevSession" in src
    assert "_chatPeekHistory" in src
    assert "_chatHistory.length = 0" in src
    # Must not wipe empty state by replaying history into #chat-msgs on load.
    restore = src.split("// Idle load:", 1)[1].split("function _chatCurrentMode", 1)[0]
    assert "_chatAppend" not in restore
    assert "empty.remove()" not in restore
    # Idle copy must not claim work already finished.
    empty = src.split('id="chat-empty"', 1)[1].split("</div>", 1)[0]
    assert "Done —" not in empty
    assert "factual Q&A pairs created" not in empty
    assert "Example (not a live result)" in empty or "not a live result" in src


def test_inline_load_context_field_is_labelled_and_shows_auto() -> None:
    """The context box beside Load had no label and showed a bare 0; empty loads with auto (parseInt -> 0)."""
    src = _src()
    field = src.split('id="chat-inline-ctx"', 1)[1].split(">", 1)[0]
    assert 'value=""' in field and 'placeholder="auto"' in field
    assert 'for="chat-inline-ctx">Context' in src
    assert "parseInt(ctxEl.value, 10) || 0" in src


def test_project_context_names_the_built_smart_brain() -> None:
    """The context table read an undefined ``rags`` and always said '0 attached' beside a ticked index (2026-10-23)."""
    src = _src()
    assert "rags" not in src
    assert "{% for idx in indexes %}{{ idx.name }} ({{ idx.doc_count }} docs)" in src and "not built yet" in src
