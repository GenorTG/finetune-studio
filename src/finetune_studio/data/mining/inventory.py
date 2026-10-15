"""A file's data-point inventory: what a complete set of pairs must state.

Two sources, merged:

1. **Rule items** (``rule_items``): every statement of every chunk that carries hard data (number, date, time, code, ALLCAPS
   identifier: ``statements.hard_tokens``). Cheap and never skips a table row, but blind to facts stated only in words.
2. **Agent items** (``agent_items``): the helper lists every data point of a chunk (and, in one whole-file pass, the facts that
   need several parts of the document, such as a later correction). Each item carries the exact ``values`` an answer must
   contain. A value that does not occur in the source text is dropped (the grounding check), so an invented "fact" never
   enters the checklist; an item left without values is dropped. An agent item that only repeats tokens a rule item of the same
   chunk already demands adds nothing and is dropped too.
"""
from __future__ import annotations

import json
import re
from collections.abc import Sequence
from dataclasses import dataclass, field

from finetune_studio.data.mining.match import contains, norm
from finetune_studio.data.mining.statements import (
    hard_tokens,
    split_statements,
    table_header_before,
)
from finetune_studio.data.prep.parsers import best_salvage
from finetune_studio.data.prep.tokens import _WORD, canon, distinctive_tokens


@dataclass
class Item:
    chunk_idx: int                 # 1-based
    origin: str                    # rule | agent | agent_file
    kind: str                      # prose | list | table | fact
    text: str
    values: list[str] = field(default_factory=list)   # rule: canonical distinctive tokens; agent: exact substrings
    header: str = ""

    def as_row(self) -> dict:
        return {"chunk_idx": self.chunk_idx, "origin": self.origin, "kind": self.kind, "text": self.text,
                "values": self.values, "header": self.header}


def rule_items(chunks: Sequence[str]) -> list[Item]:
    out: list[Item] = []
    for i, chunk in enumerate(chunks):
        for s in split_statements(chunk, table_header_before(chunks, i)):
            tokens = hard_tokens(s.text)
            if tokens:
                out.append(Item(i + 1, "rule", s.kind, s.text, sorted(tokens), s.header))
    return out


def _json_array(raw: str) -> list:
    """The first JSON array in a helper reply, tolerant of fences and prose around it; ``[]`` when unparseable."""
    text = (raw or "").strip()
    text = re.sub(r"^```(?:json)?|```$", "", text, flags=re.MULTILINE).strip()
    start, end = text.find("["), text.rfind("]")
    if start < 0 or end <= start:
        return []
    try:
        data = json.loads(text[start:end + 1])
    except json.JSONDecodeError:
        # one malformed object (a stray '",' from the helper) used to drop the WHOLE inventory of a chunk silently
        return best_salvage(text[start:end + 1])
    return data if isinstance(data, list) else []



def parse_skips(raw: str, n_items: int) -> dict[int, str]:
    """A fix reply's ``{"skip": <checklist number>, "reason": "..."}`` entries as {0-based index: reason}.

    A skip without a reason, or with a number outside the checklist, is ignored: the reviewer must see why.
    """
    out: dict[int, str] = {}
    for row in _json_array(raw):
        if not isinstance(row, dict) or "skip" not in row:
            continue
        try:
            n = int(row["skip"])
        except (TypeError, ValueError):
            continue
        reason = str(row.get("reason") or "").strip()
        if 1 <= n <= n_items and reason:
            out[n - 1] = reason
    return out


def parse_inventory(raw: str) -> list[dict]:
    """The helper's JSON array of ``{"fact", "values"}``; tolerant of fences and prose around it. ``[]`` when unparseable."""
    out = []
    for row in _json_array(raw):
        if not isinstance(row, dict):
            continue
        fact = str(row.get("fact") or "").strip()
        values = row.get("values") or []
        if isinstance(values, str):
            values = [values]
        values = [str(v).strip() for v in values if str(v).strip()]
        if fact:
            out.append({"fact": fact, "values": values})
    return out


def _token_like(value: str) -> frozenset[str] | None:
    """The value's distinctive tokens when the value is nothing BUT such tokens (a number, a code, a name), else None."""
    toks = distinctive_tokens(value)
    words = {canon(w) for w in _WORD.findall(value)}
    words = {w for w in words if len(w) > 1}
    return toks if toks and words <= toks else None


_STRUCTURE = re.compile(
    r"\b(?:section|chapter|part|heading|passage|document)\b[^.]*\b(?:is titled|is headed|is called|is named|ends with|begins with|starts with)\b"
    r"|\bthe next section\b",
    re.IGNORECASE,
)
_MAX_PHRASE_WORDS = 6


def _is_data(value: str) -> bool:
    """A value is data (number, code, name, short key words), not a sentence-long phrase without a single distinctive token."""
    return bool(distinctive_tokens(value)) or len(_WORD.findall(value)) <= _MAX_PHRASE_WORDS


def ground(entries: Sequence[dict], source_text: str) -> list[dict]:
    """Keep only data values found in ``source_text``; drop entries about document structure or left without a value."""
    out = []
    for e in entries:
        if _STRUCTURE.search(e["fact"]):
            continue
        values = [v for v in dict.fromkeys(e["values"])
                  if len(norm(v)) >= 2 and _is_data(v) and contains(v, source_text)]
        if values:
            out.append({"fact": e["fact"], "values": values})
    return out


def agent_items(entries: Sequence[dict], chunk_idx: int, rules_here: Sequence[Item], origin: str = "agent") -> list[Item]:
    """Grounded entries as items, minus the ones a rule item of the same chunk already demands."""
    rule_tokens = [frozenset(r.values) for r in rules_here]
    seen: set[tuple[str, ...]] = set()
    out: list[Item] = []
    for e in entries:
        key = tuple(sorted(norm(v) for v in e["values"]))
        if key in seen:
            continue
        seen.add(key)
        token_sets = [_token_like(v) for v in e["values"]]
        if all(t is not None for t in token_sets):
            needed = frozenset().union(*token_sets)  # type: ignore[arg-type]
            if any(needed <= rt for rt in rule_tokens):
                continue
        out.append(Item(chunk_idx, origin, "fact", e["fact"], list(e["values"])))
    return out


def chunk_of(values: Sequence[str], chunks: Sequence[str]) -> int:
    """1-based chunk that holds the first value (the whole-file pass anchors its items there)."""
    for v in values:
        for i, chunk in enumerate(chunks):
            if contains(v, chunk):
                return i + 1
    return 1


def item_prompt_text(item: dict) -> str:
    """How an item is shown to the helper in a checklist."""
    if item["origin"] == "rule":
        return f"[{item['header']}] || {item['text']}" if item.get("header") else item["text"]
    return f"{item['text']} (must state: {'; '.join(item['values'])})"
