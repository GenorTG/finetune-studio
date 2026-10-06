"""The guide's tool-calling loop as an event stream.

WHAT THIS FILE DOES
===================
``run_guide_loop`` drives one user turn: ask the model, parse tool calls, run them, feed results
back, repeat up to ``max_rounds`` — yielding an event for every step so the browser can render
tool calls the instant they happen (SSE) and the legacy JSON route can aggregate the same stream.

Guarantee: a turn that produced tool calls never ends without an assistant message. When the round
limit is hit (or the model returns nothing), one forced final-answer turn runs with tools disabled;
if even that fails, a server-built message states plainly what happened.
"""
from __future__ import annotations

import json
import logging
from collections.abc import Callable, Iterator
from typing import Any

from finetune_studio.guide.prompt import build_system_prompt
from finetune_studio.guide.toolcalls import (
    TRUNCATION_MSG,
    extract_tool_calls,
    looks_truncated,
    strip_thinking_reply,
)
from finetune_studio.guide.tools import ToolContext, run_tool

log = logging.getLogger(__name__)

FORCED_FINAL_PROMPT = (
    "You have used all your tool rounds. Answer the user NOW in plain text, using only the tool "
    "results above. Do not call any tool. If something is still unknown, say so."
)
MAX_RESULT_CHARS = 12000


def authoritative_readiness_reply(tool_calls: list[dict], reply: str) -> str:
    """Use server-computed readiness facts instead of a model paraphrase."""
    if not tool_calls or tool_calls[-1].get("name") != "inspect_project_readiness":
        return reply
    result = tool_calls[-1].get("result") or {}
    if result.get("summary"):
        return str(result["summary"])
    return reply


def _fallback_reply(tool_calls: list[dict]) -> str:
    if not tool_calls:
        return "The helper model returned no answer. Try again, or rephrase the question."
    names = ", ".join(dict.fromkeys(tc["name"] for tc in tool_calls))
    return (
        f"I ran {len(tool_calls)} tool call(s) ({names}) but the model did not write a final answer "
        "within the round limit. The results are shown above — ask me to continue or narrow the question."
    )


def run_guide_loop(
    ctx: ToolContext,
    messages: list[dict],
    chat_fn: Callable[[list[dict]], str],
    *,
    max_rounds: int = 8,
    project_name: str | None = None,
    system_prompt: str | None = None,
    max_tokens: int = 4096,
) -> Iterator[dict[str, Any]]:
    """Yield ``thinking`` / ``tool_call`` / ``tool_result`` / ``ui`` events, then ``final`` or ``error``."""
    full = [{"role": "system", "content": system_prompt or build_system_prompt(ctx, project_name)}, *messages]
    done: list[dict] = []
    seen: set[str] = set()
    rounds = 0
    reply = ""

    for _ in range(max_rounds):
        rounds += 1
        yield {"type": "thinking", "round": rounds}
        try:
            text = chat_fn(full)
        except Exception as e:
            log.exception("guide chat call failed")
            yield {"type": "error", "status": 502, "error": f"chat call failed: {e}", "rounds": rounds}
            return
        calls = extract_tool_calls(text)
        visible = strip_thinking_reply(text) or text
        if looks_truncated(text):
            yield {"type": "error", "status": 200, "truncated": True, "rounds": rounds,
                   "error": TRUNCATION_MSG.format(n=max_tokens)}
            return
        if not calls:
            reply = visible
            break
        full.append({"role": "assistant", "content": visible})
        for call in calls:
            name, args = call["name"], call.get("arguments") or {}
            index = len(done)
            yield {"type": "tool_call", "index": index, "name": name, "arguments": args}
            key = f"{name}:{json.dumps(args, sort_keys=True, default=str)}"
            if key in seen:
                result: dict[str, Any] = {"error": "identical call already made this turn; use the earlier result"}
            else:
                seen.add(key)
                result = run_tool(ctx, name, args)
            ui_event = result.pop("ui_event", None)
            record = {"name": name, "arguments": args, "result": result}
            done.append(record)
            yield {"type": "tool_result", "index": index, **record}
            if ui_event:
                yield {"type": "ui", "index": index, "event": ui_event}
            payload = json.dumps(result, ensure_ascii=False, default=str)[:MAX_RESULT_CHARS]
            full.append({"role": "user", "content": f"TOOL_RESULT {name}: {payload}"})

    forced = False
    if not reply.strip():
        forced = True
        yield {"type": "thinking", "round": rounds + 1, "forced_final": True}
        try:
            text = chat_fn([*full, {"role": "user", "content": FORCED_FINAL_PROMPT}])
            candidate = strip_thinking_reply(text) or ""
            reply = "" if extract_tool_calls(text) or looks_truncated(text) else candidate
        except Exception:
            log.exception("guide forced-final call failed")
            reply = ""
        rounds += 1
    fallback = not reply.strip()
    if fallback:
        reply = _fallback_reply(done)
    reply = authoritative_readiness_reply(done, reply)
    yield {"type": "final", "reply": reply, "rounds": rounds, "forced_final": forced, "fallback": fallback,
           "tool_calls": len(done)}
