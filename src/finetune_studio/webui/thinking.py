"""Split Qwen3-style <think>...</think> reasoning out of model responses.

Returns a dict with two fields:
  - "thinking": the reasoning block (or "" if none)
  - "response": the answer without the <think>...</think> wrapper

Supports both opening-tag-only ("<think>...rest") and full ("<think>...</think>...")
so partial streaming tokens don't leave stray opening tags.
"""

from __future__ import annotations

import re

# Match a complete <think>...</think> block (greedy across newlines, lazy inside).
_FULL_RE = re.compile(r"<think>\s*(.*?)\s*</think>\s*", re.DOTALL)
# Match an unterminated <think> opener (the rest of the output is reasoning, no answer yet).
_OPEN_RE = re.compile(r"^\s*<think>\s*(.*)$", re.DOTALL)


def split_thinking(raw: str) -> dict:
    """Pull reasoning out of `raw` if present.

    Returns:
        {"thinking": str, "response": str}

    Behavior:
      - If raw contains a complete <think>...</think> block, everything inside it
        becomes `thinking`, everything after becomes `response`.
      - If raw starts with <think> but has no closing tag, the entire content is
        treated as `thinking` (model still reasoning) and `response` is "".
      - Otherwise raw becomes `response`, `thinking` is "".
    """
    if not raw:
        return {"thinking": "", "response": ""}

    m = _FULL_RE.search(raw)
    if m:
        thinking = m.group(1).strip()
        response = (raw[: m.start()] + raw[m.end():]).strip()
        return {"thinking": thinking, "response": response}

    m = _OPEN_RE.match(raw)
    if m:
        return {"thinking": m.group(1).strip(), "response": ""}

    return {"thinking": "", "response": raw.strip()}
