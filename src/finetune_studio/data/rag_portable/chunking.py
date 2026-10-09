"""Row-preserving, token-sized chunking for the RAG index (stdlib only).

The parsers flatten spreadsheets, CSV and Word/PowerPoint tables into one text line per row
(``cell | cell | cell``). The old chunker split the whole document on whitespace and re-joined the words with spaces, which
erased every newline: the last cell of one row ended up next to the first cell of the next row, so a note ("Visit 3 Oct
10:00") read as if it belonged to the next company. This module keeps the line structure:

* every line is an atomic unit and chunks are made of whole lines joined with ``\\n``;
* a table row (a line with ``" | "`` cells, or a ``---`` separator) is **never split**, never merged into the neighbouring
  row's line, and takes no part in overlap. A row longer than the budget becomes a chunk of its own;
* when a table has a header (the CSV parser writes ``header`` + ``--- | ---``), the header is repeated at the top of each
  continuation chunk, so a row is still read with its column names;
* a prose line longer than the budget is cut on word boundaries into overlapping windows (the old behaviour, now only for
  lines that really are too long); short prose lines are packed whole, with whole lines as overlap.

Size is measured in tokens. ``count_tokens`` is the embedder's own tokenizer when the caller has one
(``get_embedder`` exposes it as ``encode.count_tokens``); without it ``estimate_tokens`` assumes
``FALLBACK_CHARS_PER_TOKEN`` characters per token. The per-line counts are summed, so a chunk's real token count differs
from the budget by a few tokens (special tokens, the joins), never by a different order of magnitude.
"""
from __future__ import annotations

import math
import re
from collections.abc import Callable
from dataclasses import dataclass

# Prose tokenises at ~4 chars/token, ids/numbers/hostnames at ~2.5-3.6. This corpus type is id-heavy, so lean small: a chunk
# that is a little shorter than the budget is harmless, one that is longer gets truncated by the embedder.
FALLBACK_CHARS_PER_TOKEN = 3.5

TokenCounter = Callable[[str], int]

_ROW_RE = re.compile(r"(^\s*\|)|(\s\|\s)")
_SEPARATOR_RE = re.compile(r"^\s*\|?\s*:?-{3,}:?\s*(\|\s*:?-{3,}:?\s*)*\|?\s*$")


def estimate_tokens(text: str) -> int:
    """Token estimate used when no tokenizer is available (``FALLBACK_CHARS_PER_TOKEN`` chars per token)."""
    return math.ceil(len(text) / FALLBACK_CHARS_PER_TOKEN) if text else 0


def is_table_row(line: str) -> bool:
    """A parser-flattened table row (``a | b``, ``| a | b |``) or its ``--- | ---`` header separator."""
    return bool(_ROW_RE.search(line)) or bool(_SEPARATOR_RE.match(line) and "|" in line)


def is_table_separator(line: str) -> bool:
    return bool(_SEPARATOR_RE.match(line))


@dataclass(frozen=True)
class _Unit:
    text: str
    tokens: int
    row: bool = False
    table: int = -1          # id of the table block a row belongs to
    first_of_table: bool = False


def _split_long_line(line: str, size: int, overlap: int, count: TokenCounter) -> list[str]:
    """Word-boundary windows of at most ``size`` tokens, consecutive windows sharing up to ``overlap`` tokens."""
    words = line.split()
    costs = [max(1, count(w)) for w in words]
    pieces: list[str] = []
    start = 0
    while start < len(words):
        end, used = start, 0
        while end < len(words) and (end == start or used + costs[end] <= size):
            used += costs[end]
            end += 1
        pieces.append(" ".join(words[start:end]))
        if end >= len(words):
            break
        back, shared = end, 0
        while back > start + 1 and shared + costs[back - 1] <= overlap:
            back -= 1
            shared += costs[back]
        start = back if back > start else end
    return pieces


def _build_units(lines: list[str], size: int, overlap: int, count: TokenCounter) -> tuple[list[_Unit], dict[int, tuple[_Unit, ...]]]:
    """Lines -> atomic units, plus each table's header units (header line + separator) keyed by table id."""
    units: list[_Unit] = []
    headers: dict[int, tuple[_Unit, ...]] = {}
    table = -1
    prev_row = False
    for i, raw in enumerate(lines):
        line = raw.rstrip()
        if not line.strip():
            if units and units[-1].text:          # one blank line = paragraph break; runs of blanks collapse
                units.append(_Unit("", 0))
            prev_row = False
            continue
        if is_table_row(line):
            first = not prev_row
            if first:
                table += 1
            unit = _Unit(line, max(1, count(line)), row=True, table=table, first_of_table=first)
            units.append(unit)
            if first and i + 1 < len(lines) and is_table_separator(lines[i + 1].rstrip()):
                headers[table] = (unit, _Unit(lines[i + 1].rstrip(), max(1, count(lines[i + 1].rstrip())), row=True, table=table))
            prev_row = True
            continue
        prev_row = False
        tokens = max(1, count(line))
        if tokens <= size:
            units.append(_Unit(line, tokens))
        else:
            units.extend(_Unit(p, max(1, count(p))) for p in _split_long_line(line, size, overlap, count))
    return units, headers


def _join(parts: list[_Unit]) -> str:
    return "\n".join(u.text for u in parts).strip("\n")


def chunk_rows(text: str, size: int, overlap: int, count_tokens: TokenCounter | None = None) -> list[str]:
    """Split ``text`` into chunks of whole lines (see the module docstring). ``size``/``overlap`` are in tokens."""
    if not text.strip():
        return []
    size = max(1, int(size))
    overlap = max(0, min(int(overlap), size - 1))
    count = count_tokens or estimate_tokens
    lines = text.replace("\r\n", "\n").replace("\r", "\n").split("\n")
    units, headers = _build_units(lines, size, overlap, count)

    chunks: list[str] = []
    cur: list[_Unit] = []
    cur_tokens = 0
    fresh = 0            # leading units of ``cur`` that are only a header / overlap repeat (not new content)

    def close() -> None:
        body = _join(cur)
        if body.strip():
            chunks.append(body)

    def restart(nxt: _Unit) -> None:
        """Seed the next chunk: the table header for a row mid-table, else whole trailing prose lines as overlap."""
        nonlocal cur, cur_tokens, fresh
        seed: list[_Unit] = []
        if nxt.row:
            head = headers.get(nxt.table)
            if head and not nxt.first_of_table and nxt is not head[1]:
                seed = list(head)
        elif overlap and cur and not cur[-1].row:
            taken = 0
            for u in reversed(cur):
                if u.row or taken + u.tokens > overlap:
                    break
                seed.insert(0, u)
                taken += u.tokens
            while seed and not seed[0].text:
                seed.pop(0)
        seed_tokens = sum(u.tokens for u in seed)
        if not seed or seed_tokens + nxt.tokens > size:
            seed, seed_tokens = [], 0
        cur, cur_tokens, fresh = seed, seed_tokens, len(seed)

    for unit in units:
        if len(cur) > fresh and cur_tokens + unit.tokens > size:
            close()
            restart(unit)
        if not cur and not unit.text:
            continue                                  # no blank line at the top of a chunk
        cur.append(unit)
        cur_tokens += unit.tokens
    if len(cur) > fresh:
        close()
    return chunks
