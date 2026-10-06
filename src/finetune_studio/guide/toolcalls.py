"""Parsing of model replies into tool calls (shared by the guide loop and the legacy route).

Local GGUF helpers emit ``<tool_call>{...}</tool_call>`` text; this module strips Qwen-style
thinking blocks and extracts the calls robustly (closed tag first, brace-balanced fallback for
an unclosed tag). Moved verbatim from ``webui/routes/data_prep_chat.py``, which re-exports the
old underscore names for existing callers and tests.
"""
from __future__ import annotations

import json
import re

TOOL_CALL_RE = re.compile(r"<tool_call>\s*(\{.*?\})\s*</tool_call>", re.DOTALL)


def strip_thinking(text: str) -> str:
    """Remove Qwen3 thinking blocks from model output.

    Handles both paired ``<think>…</think>`` and the common chat-template
    leak where only a bare closing ``</think>`` appears (opening tag was
    injected into the prompt, so the model never emits it).
    """
    cleaned = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL)
    if "</think>" in cleaned:
        cleaned = cleaned.split("</think>", 1)[1]
    return cleaned


def extract_tool_calls(text: str) -> list[dict]:
    """Pull `<tool_call>{...}</tool_call>` blocks out of a model reply.

    Robust against two recurring issues with local Qwen3 GGUF + llama-cpp:
      1. The model leaks chain-of-thought (Qwen3's native thinking-mode
         output). Strip `<think>...</think>` blocks BEFORE regex matching so
         they don't contaminate the tool-call JSON or appear in the visible
         reply. Also drop everything up to a bare ``</think>``.
      2. The model frequently emits `<tool_call>{...}` WITHOUT a closing
         `</tool_call>` tag. Try the strict closed form first; on miss,
         fall back to a brace-balanced extractor that walks the unmatched
         opening tag and grabs everything up to the first balanced `}`.
    """
    # 1. Strip Qwen3 thinking-mode blocks (paired + bare closing tag).
    cleaned = strip_thinking(text)
    # 2. Strip any leading/trailing prose so the closing tag (or unclosed
    # block) is clearly delimited. We do NOT mutate the text that flows
    # into the visible reply (that's `strip_thinking_reply`'s job).
    calls: list[dict] = []
    seen: set[tuple[str, str]] = set()  # dedupe by (name, json_args)

    def _try_parse(raw: str) -> None:
        try:
            obj = json.loads(raw)
        except json.JSONDecodeError:
            return
        name = obj.get("name")
        args = obj.get("arguments") or {}
        if not isinstance(name, str) or not isinstance(args, dict):
            return
        key = (name, json.dumps(args, sort_keys=True))
        if key in seen:
            return
        seen.add(key)
        calls.append({"name": name, "arguments": args})

    # 2a. Strict pass: properly-closed <tool_call>{...}</tool_call>.
    for m in TOOL_CALL_RE.finditer(cleaned):
        _try_parse(m.group(1))

    # 2b. Fallback pass: unclosed <tool_call>{...}  (Qwen3 occasionally
    # forgets the closing tag). Find every opening tag, then brace-balance
    # forward to find the end of the JSON object.
    if not calls:
        for m in re.finditer(r"<tool_call>\s*", cleaned):
            start = m.end()
            depth = 0
            in_string = False
            escape = False
            end = -1
            for i in range(start, len(cleaned)):
                ch = cleaned[i]
                if escape:
                    escape = False
                    continue
                if ch == "\\":
                    escape = True
                    continue
                if ch == '"':
                    in_string = not in_string
                    continue
                if in_string:
                    continue
                if ch == "{":
                    depth += 1
                elif ch == "}":
                    depth -= 1
                    if depth == 0:
                        end = i + 1
                        break
            if end > start:
                _try_parse(cleaned[start:end])
            # Stop after the first successful extraction — the parser
            # driver already enforces one-tool-call-per-round upstream.
            if calls:
                break

    return calls


def strip_thinking_reply(text: str) -> str:
    """Strip thinking blocks from a model reply before it becomes the
    user-visible assistant message."""
    return strip_thinking(text).strip()


TRUNCATION_MSG = (
    "Response was cut off at max_tokens={n} — raise Max tokens and retry"
)


def looks_truncated(text: str) -> bool:
    """True when generation likely hit max_tokens mid-tool-call or think."""
    if not text:
        return False
    # Unclosed think block (opening present, no closer).
    if "<think>" in text and "</think>" not in text:
        return True
    # Truncated tool call: opening tag present but we couldn't parse a call.
    return "<tool_call>" in text and not extract_tool_calls(text)
