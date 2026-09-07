"""Parse raw model output into a list of {"q":..., "a":...} dicts.

Order of attempts (each step is a fallback):
  1. Strip <think>...</think> blocks (Qwen3 emits these by default).
  2. Strip ```json ... ``` markdown fences.
  3. Try each `[` position; the first one whose full JSON parse succeeds wins.
  4. Last resort: numbered-list / Q:/A: regex extraction.
"""
from __future__ import annotations

import json
import re
from typing import Optional


def find_matching_bracket(s: str, start: int) -> Optional[int]:
    """Return the index of the ']' that balances the '[' at `start`. String-aware."""
    if start >= len(s) or s[start] != "[":
        return None
    depth = 0
    in_str = False
    escape = False
    for i in range(start, len(s)):
        ch = s[i]
        if escape:
            escape = False
            continue
        if ch == "\\" and in_str:
            escape = True
            continue
        if ch == '"':
            in_str = not in_str
            continue
        if in_str:
            continue
        if ch == "[":
            depth += 1
        elif ch == "]":
            depth -= 1
            if depth == 0:
                return i
    return None


def coerce_pairs(arr, n_expected: int) -> list[dict]:
    """Accept whatever the model returned; normalise to [{"q":..., "a":...}]."""
    out = []
    if isinstance(arr, list):
        for it in arr:
            if isinstance(it, dict):
                q = (it.get("q") or it.get("question") or "").strip()
                a = (it.get("a") or it.get("answer") or "").strip()
                if q and a:
                    out.append({"q": q, "a": a})
    return out[:n_expected]


def parse_qa_json(raw: str, n_expected: int) -> list[dict]:
    """The main entry point. Tries each fallback in turn."""
    if not raw:
        return []
    s = raw.strip()

    # 1. Strip thinking/reasoning blocks (Qwen3 emits these by default)
    s = re.sub(r"<think>.*?</think>", "", s, flags=re.DOTALL).strip()
    # Also strip unterminated trailing <think>... blocks (model forgot to close)
    s = re.sub(r"<think>.*$", "", s, flags=re.DOTALL).strip()

    # 2. Strip markdown fences
    fence = re.search(r"```(?:json)?\s*(\[.*?\]|\{.*?\})\s*```", s, re.DOTALL)
    if fence:
        s = fence.group(1)

    # 3. Try to find a JSON array whose content parses cleanly.
    # Try each candidate `[` position; pick the first whose full parse succeeds.
    candidate_starts = [i for i, ch in enumerate(s) if ch == "["]
    for start in candidate_starts:
        end = find_matching_bracket(s, start)
        if end is None or end <= start:
            continue
        candidate = s[start:end + 1]
        for variant in (candidate, candidate.replace("'", '"')):
            try:
                arr = json.loads(variant)
                return coerce_pairs(arr, n_expected)
            except Exception:
                continue

    # 4. Last resort: numbered-list / Q:/A: extraction
    return parse_qa_lines(s, n_expected)


def parse_qa_lines(s: str, n_expected: int) -> list[dict]:
    """Last-resort: extract Q/A pairs from numbered lists like '1 Q: ... A: ...'."""
    out = []
    pattern = re.compile(
        r"(?:^|\n)\s*(?:\d+[\.)]\s*|[-*]\s*)?Q[:\.\)]?\s*(.+?)\s*(?:A[:\.\)]?\s*(.+?))(?=\n\s*(?:\d+[\.)]|[-*]\s*Q|$))",
        re.IGNORECASE | re.DOTALL,
    )
    for m in pattern.finditer(s):
        q, a = m.group(1).strip(), m.group(2).strip()
        if q and a:
            out.append({"q": q, "a": a})
            if len(out) >= n_expected:
                break
    return out[:n_expected]
