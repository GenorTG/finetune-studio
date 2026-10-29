"""Guide loop + SSE route with a scripted fake helper (no model, no GPU)."""
from __future__ import annotations

import json
from collections.abc import Callable
from typing import Any

import pytest

from finetune_studio.guide import loop as guide_loop
from finetune_studio.guide.loop import FORCED_FINAL_PROMPT, run_guide_loop
from finetune_studio.guide.tools import ToolContext


def call(name: str, **arguments: Any) -> str:
    return f'<tool_call>{json.dumps({"name": name, "arguments": arguments})}</tool_call>'


class Script:
    """Fake helper: replays canned replies in order and records every prompt it was given."""

    def __init__(self, *replies: str | Callable[[list[dict]], str]) -> None:
        self.replies = list(replies)
        self.seen: list[list[dict]] = []

    def __call__(self, messages: list[dict]) -> str:
        self.seen.append(messages)
        reply = self.replies[min(len(self.seen), len(self.replies)) - 1]
        return reply(messages) if callable(reply) else reply


def types(events: list[dict]) -> list[str]:
    return [e["type"] for e in events]


USER = [{"role": "user", "content": "how do I train?"}]
CTX = ToolContext(pid=None, page="dashboard")


def test_loop_streams_tool_call_then_result_then_final() -> None:
    chat = Script(call("app_help", query="how do I train"), "Open the Training page and pick a preset.")
    events = list(run_guide_loop(CTX, USER, chat))
    assert types(events) == ["thinking", "tool_call", "tool_result", "thinking", "final"]
    result = next(e for e in events if e["type"] == "tool_result")
    assert result["name"] == "app_help" and result["result"]["results"]
    assert events[-1]["reply"].startswith("Open the Training page")
    assert events[-1]["forced_final"] is False and events[-1]["fallback"] is False
    # the model saw the tool result on its second turn
    assert any("TOOL_RESULT app_help" in m["content"] for m in chat.seen[1])
    # and the system prompt carries the live page context + the generated tool list
    system = chat.seen[0][0]["content"]
    assert "Dashboard" in system and "- app_help(query)" in system and "suggest_settings(page, settings)" in system


def test_prompt_includes_fresh_project_readiness_over_old_chat_state(monkeypatch: pytest.MonkeyPatch) -> None:
    from finetune_studio.guide import state
    from finetune_studio.guide.prompt import build_system_prompt

    summary = "1 source(s), 1 parsed; 0 approved, 28 pending, and 0 rejected Q&A pair(s). Next: Review pending Q&A pairs."
    monkeypatch.setattr(state, "readiness", lambda _pid: {"summary": summary})

    prompt = build_system_prompt(ToolContext(pid="project-1", page="data-prep"), "scratch-project")

    assert "# Current project state" in prompt
    assert "overrides older project-state/tool results in chat history" in prompt
    assert summary in prompt


def test_prompt_includes_authoritative_current_page_kb_entry() -> None:
    from finetune_studio.guide.prompt import build_system_prompt

    prompt = build_system_prompt(ToolContext(pid="project-1", page="project_settings"))

    assert "# Current page reference" in prompt
    assert "Change settings for this project only" in prompt
    assert "The folded WebUI log tail below the form is the studio service log shared across projects." in prompt
    assert "Do not infer controls or behavior that this reference does not list." in prompt


def test_prompt_prefers_page_specific_entry_over_generic_app_map() -> None:
    from finetune_studio.guide.prompt import build_system_prompt

    prompt = build_system_prompt(ToolContext(pid=None, page="dashboard"))

    assert "Dashboard (page: /)" in prompt
    assert "The Dashboard is the studio-wide start page." in prompt
    assert "One-page shortcut: Quick work" not in prompt
    assert "For informational questions, answer every part in place" in prompt
    assert "never navigate unless the user explicitly asks to open or switch pages" in prompt
    assert "Never substitute an action for the answer" in prompt


def test_training_prompt_uses_step_two_from_the_project_flow() -> None:
    from finetune_studio.guide.prompt import build_system_prompt

    prompt = build_system_prompt(ToolContext(pid="project-1", page="training"))

    assert "Training page (step 2)" in prompt
    assert "Step 2 of the project flow" in prompt
    assert "Training page (step 3)" not in prompt
    assert "Training loss measures fit to the training examples; it does not prove held-out quality." in prompt
    assert "Final loss below 0.5" not in prompt


@pytest.mark.parametrize(
    ("page", "expected_title", "stale_title"),
    [
        ("pairs", "Question-answer pairs page (step 1)", "Question-answer pairs page (step 2)"),
        ("testing", "Testing page (step 3)", "Testing page (step 4)"),
        ("export", "Export page (step 4)", "Export page (step 5)"),
    ],
)
def test_project_flow_page_references_match_the_visible_navigation(
    page: str, expected_title: str, stale_title: str,
) -> None:
    from finetune_studio.guide.prompt import build_system_prompt

    prompt = build_system_prompt(ToolContext(pid="project-1", page=page))

    assert expected_title in prompt
    assert stale_title not in prompt


def test_settings_prompt_states_the_real_claude_and_heretic_paths() -> None:
    from finetune_studio.guide.kb import load_kb
    from finetune_studio.guide.prompt import (
        _CURRENT_PAGE_HELP_MAX_CHARS,
        build_system_prompt,
    )

    settings_entry = load_kb()["settings"]
    full_reference = settings_entry.render(max_chars=100_000)
    prompt = build_system_prompt(ToolContext(pid=None, page="settings"))

    assert len(full_reference) <= _CURRENT_PAGE_HELP_MAX_CHARS
    assert full_reference in prompt

    assert "OpenRouter" in prompt
    assert "anthropic/claude-3.5-sonnet" in prompt
    assert "This form does not have an Anthropic-specific preset." in prompt
    assert "Heretic (integrated optional feature)" in prompt
    assert "The Heretic format remains an Export-page feature." in prompt


def test_prompt_survives_unavailable_project_readiness(monkeypatch: pytest.MonkeyPatch) -> None:
    from finetune_studio.guide import state
    from finetune_studio.guide.prompt import build_system_prompt

    def unavailable(_pid: str) -> dict:
        raise RuntimeError("database unavailable")

    monkeypatch.setattr(state, "readiness", unavailable)
    prompt = build_system_prompt(ToolContext(pid="project-1", page="data-prep"))

    assert "# Current project state" in prompt
    assert "snapshot could not be read this turn" in prompt


def test_tool_call_event_precedes_its_result_so_the_ui_shows_it_immediately() -> None:
    messages = [{"role": "user", "content": "Open Settings."}]
    chat = Script(call("navigate", page="settings"), "Opened Settings.")
    events = list(run_guide_loop(CTX, messages, chat))
    names = types(events)
    assert names.index("tool_call") < names.index("tool_result") < names.index("ui")
    ui = next(e for e in events if e["type"] == "ui")
    assert ui["event"] == {"type": "navigate", "page": "settings", "url": "/settings", "title": "Settings"}
    # the ui event is NOT left inside the stored result (and not fed back to the model)
    assert "ui_event" not in next(e for e in events if e["type"] == "tool_result")["result"]


def test_informational_question_cannot_switch_the_page_without_permission(monkeypatch: pytest.MonkeyPatch) -> None:
    executed: list[str] = []
    monkeypatch.setattr(guide_loop, "run_tool", lambda ctx, name, args: executed.append(name) or {})
    messages = [{"role": "user", "content": "What does the Testing page do?"}]
    chat = Script(call("navigate", page="pairs"), "Testing runs answers first; judging scores them afterwards.")

    events = list(run_guide_loop(ToolContext(pid="project-1", page="testing"), messages, chat))

    navigation = next(e for e in events if e["type"] == "tool_result")
    assert navigation["result"]["error"].startswith("The user asked for information")
    assert "ui" not in types(events)
    assert executed == []
    assert events[-1]["reply"] == "Testing runs answers first; judging scores them afterwards."


def test_round_limit_forces_a_final_answer() -> None:
    """Live finding: a tool loop that used every round ended with no assistant message."""
    n = {"i": 0}

    def next_call(_m: list[dict]) -> str:
        n["i"] += 1
        return call("app_help", query=f"topic {n['i']}")

    chat = Script(next_call, next_call, next_call, "Based on the lookups: pick Baseline and press Start.")
    # the 4th model call is the forced one; calls 1-3 are the rounds
    events = list(run_guide_loop(CTX, USER, chat, max_rounds=3))
    final = events[-1]
    assert final["type"] == "final" and final["forced_final"] is True and final["fallback"] is False
    assert final["reply"] == "Based on the lookups: pick Baseline and press Start."
    assert len(chat.seen) == 4
    assert chat.seen[-1][-1] == {"role": "user", "content": FORCED_FINAL_PROMPT}
    assert types(events).count("tool_result") == 3
    assert any(e.get("forced_final") for e in events if e["type"] == "thinking")


def test_forced_final_that_still_calls_a_tool_falls_back_to_an_honest_message() -> None:
    n = {"i": 0}

    def always_tool(_m: list[dict]) -> str:
        n["i"] += 1
        return call("app_help", query=f"q{n['i']}")

    events = list(run_guide_loop(CTX, USER, Script(always_tool), max_rounds=2))
    final = events[-1]
    assert final["type"] == "final" and final["fallback"] is True and final["forced_final"] is True
    assert "app_help" in final["reply"] and "did not write a final answer" in final["reply"]


def test_empty_reply_gets_a_forced_turn_not_silence() -> None:
    events = list(run_guide_loop(CTX, USER, Script("", "Here is the answer.")))
    final = events[-1]
    assert final["reply"] == "Here is the answer." and final["forced_final"] is True


def test_empty_everywhere_still_yields_a_message() -> None:
    final = list(run_guide_loop(CTX, USER, Script("")))[-1]
    assert final["type"] == "final" and final["fallback"] is True and final["reply"].strip()


def test_identical_repeat_call_is_not_executed_twice(monkeypatch: pytest.MonkeyPatch) -> None:
    ran: list[str] = []
    monkeypatch.setattr(guide_loop, "run_tool", lambda ctx, name, args: ran.append(name) or {"written": 1})
    chat = Script(call("create_qa_pairs", source_id="s", pairs=[]), call("create_qa_pairs", source_id="s", pairs=[]), "Done.")
    events = list(run_guide_loop(CTX, USER, chat))
    results = [e["result"] for e in events if e["type"] == "tool_result"]
    assert ran == ["create_qa_pairs"]
    assert results[1]["error"].startswith("identical call already made")


def test_readiness_summary_is_appended_without_replacing_the_answer(monkeypatch: pytest.MonkeyPatch) -> None:
    summary = "2 source(s), 1 parsed; 3 approved, 3 pending, and 0 rejected Q&A pair(s); 1 dataset(s); 0 RAG corpus/corpora. Next: Review pending Q&A pairs."
    monkeypatch.setattr(guide_loop, "run_tool", lambda ctx, name, args: {"summary": summary})
    chat = Script(call("inspect_project_readiness"), "Testing is step 3. Run needs an exported model.")
    final = list(run_guide_loop(CTX, USER, chat))[-1]
    assert final["reply"].startswith("Testing is step 3. Run needs an exported model.")
    assert f"Current project state: {summary}" in final["reply"]


def test_readiness_summary_is_not_duplicated_when_the_answer_includes_it() -> None:
    summary = "1 source(s), 1 parsed; 0 approved pairs."
    tool_calls = [{"name": "inspect_project_readiness", "result": {"summary": summary}}]

    assert guide_loop.authoritative_readiness_reply(tool_calls, f"State: {summary}") == f"State: {summary}"


def test_truncated_tool_call_is_reported_not_swallowed() -> None:
    events = list(run_guide_loop(CTX, USER, Script('<tool_call>{"name":"app_help","arguments":{"query":"x'), max_tokens=512))
    assert events[-1]["type"] == "error" and events[-1]["truncated"] and "max_tokens=512" in events[-1]["error"]


def test_model_failure_is_a_502_error_event() -> None:
    def boom(_m: list[dict]) -> str:
        raise RuntimeError("helper crashed")

    last = list(run_guide_loop(CTX, USER, boom))[-1]
    assert last["type"] == "error" and last["status"] == 502 and "helper crashed" in last["error"]


def test_thinking_blocks_never_reach_the_reply() -> None:
    final = list(run_guide_loop(CTX, USER, Script("<think>secret plan</think>Use Baseline.")))[-1]
    assert final["reply"] == "Use Baseline."


# ── HTTP: SSE route + the legacy JSON route ─────────────────────────────

EXTERNAL = {"base_url": "http://stub", "api_key": "k", "model_id": "stub"}


def sse_events(response: Any) -> list[dict]:
    return [json.loads(line[6:]) for line in response.text.splitlines() if line.startswith("data: ")]


@pytest.fixture
def scripted(monkeypatch: pytest.MonkeyPatch) -> Callable[..., Script]:
    def install(*replies: str | Callable[[list[dict]], str]) -> Script:
        script = Script(*replies)
        monkeypatch.setattr("finetune_studio.webui.chat_backend._chat_external", lambda backend, msgs, gen=None: script(msgs))
        return script

    return install


def test_sse_route_streams_events_in_order(client, scripted) -> None:
    from finetune_studio import db

    project = db.create_project(name="Guide SSE")
    pid = project["id"] if isinstance(project, dict) else project
    scripted(call("navigate", page="training"), "I opened Training.")
    response = client.post("/api/guide/chat", json={
        "messages": [{"role": "user", "content": "take me to training"}],
        "project_id": pid, "page_path": f"/projects/{pid}/data", "external_api": EXTERNAL,
    })
    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/event-stream")
    events = sse_events(response)
    assert types(events) == ["start", "thinking", "tool_call", "tool_result", "ui", "thinking", "final", "done"]
    assert events[0]["page"] == "files"
    assert events[0]["project_id"] == pid
    assert events[4]["event"]["url"] == f"/projects/{pid}/training"
    assert events[-2]["reply"] == "I opened Training."


def test_sse_route_forces_a_final_answer_at_the_round_limit(client, scripted) -> None:
    n = {"i": 0}

    def tool(_m: list[dict]) -> str:
        n["i"] += 1
        return call("app_help", query=f"topic {n['i']}")

    scripted(tool, tool, "Short answer after the lookups.")
    response = client.post("/api/guide/chat", json={
        "messages": [{"role": "user", "content": "explain everything"}], "external_api": EXTERNAL, "max_rounds": 2,
    })
    events = sse_events(response)
    final = next(e for e in events if e["type"] == "final")
    assert final["forced_final"] and final["reply"] == "Short answer after the lookups."


def test_sse_route_validates_input(client) -> None:
    assert client.post("/api/guide/chat", json={}).status_code == 400
    assert client.post("/api/guide/chat", json={"messages": [{"role": "assistant", "content": "hi"}], "external_api": EXTERNAL}).status_code == 400
    assert client.post("/api/guide/chat", json={"messages": [{"role": "user", "content": "x"}], "external_api": "no"}).status_code == 400
    missing = client.post("/api/guide/chat", json={"messages": [{"role": "user", "content": "x"}], "project_id": "nope0000", "external_api": EXTERNAL})
    assert missing.status_code == 404


def test_sse_route_without_a_usable_assistant_is_a_clear_error_never_a_silent_switch(client, monkeypatch: pytest.MonkeyPatch) -> None:
    """The Guide runs on the guide job's connection; when that connection cannot be used, the stream ends with an
    error naming it, and no other loaded model answers instead."""
    monkeypatch.setattr("finetune_studio.models.helper.get_helper_provider_id", lambda *_a, **_k: "gone-connection")
    response = client.post("/api/guide/chat", json={"messages": [{"role": "user", "content": "hi"}]})
    events = sse_events(response)
    errors = [e for e in events if e["type"] == "error"]
    assert errors and "gone-connection" in errors[0]["error"]
    assert not any(e["type"] == "final" for e in events)


def test_kb_lookup_and_search() -> None:
    from finetune_studio.guide.kb import load_kb, table_of_contents
    from finetune_studio.guide.search import search_kb

    assert any(e["id"] == "training" for e in table_of_contents())
    entry = load_kb()["training"]
    assert entry.route.endswith("/training") and "Workflow" in entry.sections
    assert "nope" not in load_kb()
    assert search_kb("which gguf quant")[0]["entry"] == "export"


def test_local_helper_failure_surfaces_as_an_error_not_an_empty_reply() -> None:
    """A dead helper used to be swallowed into '' (then 'no answer'); the guide must say why."""
    from unittest.mock import MagicMock

    from finetune_studio.webui.chat_backend import make_chat_fn

    mgr = MagicMock()
    mgr.chat.side_effect = RuntimeError("500 Server Error: helper down")
    mgr.generate.side_effect = RuntimeError("fallback also down")
    chat = make_chat_fn({"kind": "provider", "manager": mgr}, {})
    last = list(run_guide_loop(CTX, USER, chat))[-1]
    assert last["type"] == "error" and "500 Server Error: helper down" in last["error"]
