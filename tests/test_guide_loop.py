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


def test_tool_call_event_precedes_its_result_so_the_ui_shows_it_immediately() -> None:
    chat = Script(call("navigate", page="settings"), "Opened Settings.")
    events = list(run_guide_loop(CTX, USER, chat))
    names = types(events)
    assert names.index("tool_call") < names.index("tool_result") < names.index("ui")
    ui = next(e for e in events if e["type"] == "ui")
    assert ui["event"] == {"type": "navigate", "page": "settings", "url": "/settings", "title": "Settings"}
    # the ui event is NOT left inside the stored result (and not fed back to the model)
    assert "ui_event" not in next(e for e in events if e["type"] == "tool_result")["result"]


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


def test_readiness_summary_is_authoritative(monkeypatch: pytest.MonkeyPatch) -> None:
    summary = "2 source(s), 1 parsed; 3 approved, 3 pending, and 0 rejected Q&A pair(s); 1 dataset(s); 0 RAG corpus/corpora. Next: Review pending Q&A pairs."
    monkeypatch.setattr(guide_loop, "run_tool", lambda ctx, name, args: {"summary": summary})
    chat = Script(call("inspect_project_readiness"), "You have 7 approved pairs and 9 sources!")
    final = list(run_guide_loop(CTX, USER, chat))[-1]
    assert final["reply"] == summary


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
        monkeypatch.setattr("finetune_studio.webui.routes.data_prep_chat._chat_external", lambda backend, msgs, gen=None: script(msgs))
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


def test_sse_route_without_a_helper_is_a_clear_409_never_a_silent_switch(client, monkeypatch: pytest.MonkeyPatch) -> None:
    from unittest.mock import MagicMock

    from finetune_studio.data.prep.generator import HELPER_NO_MODEL_MSG

    mgr = MagicMock()
    mgr.active.return_value = None
    engine = MagicMock()
    engine.model = None
    monkeypatch.setattr("finetune_studio.models.manager.get_manager", lambda: mgr)
    monkeypatch.setattr("finetune_studio.webui.app.inference_engine", engine)
    response = client.post("/api/guide/chat", json={"messages": [{"role": "user", "content": "hi"}]})
    assert response.status_code == 409
    assert response.json()["error"] == HELPER_NO_MODEL_MSG


def test_kb_endpoints(client) -> None:
    toc = client.get("/api/guide/kb").json()["entries"]
    assert any(e["id"] == "training" for e in toc)
    entry = client.get("/api/guide/kb/training").json()
    assert entry["page"].endswith("/training") and "Workflow" in entry["sections"]
    assert client.get("/api/guide/kb/nope").status_code == 404
    hits = client.get("/api/guide/help", params={"q": "which gguf quant"}).json()["results"]
    assert hits[0]["entry"] == "export"
    assert client.get("/api/guide/help", params={"q": " "}).status_code == 400


def test_legacy_json_route_also_ends_with_a_message_at_the_round_limit(client, scripted) -> None:
    """The route the old UI used: 6 tool rounds used to end with an empty reply."""
    from finetune_studio import db

    project = db.create_project(name="Legacy JSON")
    pid = project["id"] if isinstance(project, dict) else project
    n = {"i": 0}

    def tool(_m: list[dict]) -> str:
        n["i"] += 1
        return call("list_sources") if n["i"] % 2 else call("app_help", query=f"q{n['i']}")

    scripted(*([tool] * 6), "Wrapped up.")
    body = client.post(f"/api/projects/{pid}/data-prep/chat", json={
        "messages": [{"role": "user", "content": "go"}], "external_api": EXTERNAL, "max_rounds": 6,
    }).json()
    assert body["ok"] and body["reply"] == "Wrapped up." and body["forced_final"] is True
    assert len(body["tool_calls"]) == 6
