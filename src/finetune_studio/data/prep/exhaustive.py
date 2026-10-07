"""Exhaustive mining: every statement of a chunk must end up in a Q&A pair, and the app can prove it.

The sampled miner asked the helper for ``n`` pairs per chunk and called a chunk "covered" when one pair survived. A 1,200-char
chunk of a rate table or a policy holds 8–15 checkable facts, so most of a document never reached the dataset (measured on a
hand-written corpus: 42 % of facts covered, tables 4–15 %). This module replaces "n pairs per chunk" with a loop that runs to
completion and is verified without any model:

  1. ``split_statements`` cuts a chunk into statements (a table row, a bullet, a sentence), each with its *distinctive
     tokens*: numbers, dates, codes, capitalised names — the things a reader could be asked about.
  2. Round 0 asks the helper for one pair per distinct fact (no cap), at temperature 0 so a re-run gives the same pairs.
  3. ``uncovered`` checks which statements' distinctive tokens are missing from every accepted answer. Those statements go
     back to the helper in gap rounds ("cover exactly these").
  4. What is still uncovered gets a deterministic extractive pair (table rows keep their header names), flagged
     ``origin="extractive_gap"`` so the reviewer sees it was not model-written.

Coverage is therefore a measured number (statements covered / statements with facts), not "at least one pair".
"""
from __future__ import annotations

import re
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field

from finetune_studio.data.prep.coverage_fill import split_sentences
from finetune_studio.data.prep.tokens import _WORD, canon, distinctive_tokens

MAX_GAP_ROUNDS = 3
# A statement counts as covered when ALL its distinctive tokens appear in the relevant answers (a phone number or a
# certificate id is exactly the token a "85 % is enough" rule drops); long statements may miss up to 10 % of theirs.
COVER_THRESHOLD = 0.9
STRICT_UP_TO = 8
# Statements longer than this are cut at ';' / ' — ' so one statement is one or two facts, not a paragraph.
MAX_STATEMENT_CHARS = 420

@dataclass(frozen=True)
class Statement:
    idx: int
    text: str
    kind: str                      # prose | table | list
    tokens: frozenset[str]         # distinctive, canonicalised
    header: str = ""               # table rows: the header row that names the columns

    @property
    def prompt_text(self) -> str:
        return f"[{self.header}] || {self.text}" if self.header else self.text


def _is_table_line(line: str) -> bool:
    return line.count(" | ") >= 1


def split_statements(chunk: str, carried_header: str = "") -> list[Statement]:
    """Cut a chunk into statements; table rows carry their header (from this chunk or ``carried_header``)."""
    statements: list[Statement] = []
    header = ""
    prev_table = False
    seen_content = False
    for raw in chunk.splitlines():
        line = raw.strip()
        if not line:
            prev_table = False
            continue
        if line.startswith("==="):          # "=== Sheet: x ===" / "=== Slide 3 ===" markers reset table context
            header, prev_table, seen_content = "", False, True
            continue
        if _is_table_line(line):
            if not prev_table:
                if carried_header and not seen_content:
                    header = carried_header  # the chunk starts mid-table: its rows belong to the earlier header
                else:
                    header = line            # a new table block: its first line names the columns
                    statements.append(Statement(len(statements), line, "table", distinctive_tokens(line), ""))
                    prev_table = seen_content = True
                    continue
            prev_table = seen_content = True
            statements.append(Statement(len(statements), line, "table", distinctive_tokens(line), header))
            continue
        prev_table = False
        seen_content = True
        kind = "list" if re.match(r"^([-*•]|\d+[.)])\s+", line) else "prose"
        pieces = [line] if kind == "list" else split_sentences(line)
        for piece in pieces:
            for part in _cut_long(piece):
                if len(part.strip()) >= 12:
                    statements.append(Statement(len(statements), part.strip(), kind, distinctive_tokens(part)))
    return statements


def _cut_long(text: str) -> list[str]:
    if len(text) <= MAX_STATEMENT_CHARS:
        return [text]
    parts = re.split(r";\s+|\s+[—–]\s+", text)
    out, cur = [], ""
    for p in parts:
        if cur and len(cur) + len(p) > MAX_STATEMENT_CHARS:
            out.append(cur)
            cur = p
        else:
            cur = f"{cur}; {p}" if cur else p
    if cur:
        out.append(cur)
    return out


def table_header_before(chunks: Sequence[str], idx: int) -> str:
    """Header row of the table a chunk continues (scan earlier chunks back to the last table-block start)."""
    for prev in range(idx - 1, -1, -1):
        lines = [ln.strip() for ln in chunks[prev].splitlines() if ln.strip()]
        header = ""
        run = 0
        for ln in lines:
            if ln.startswith("==="):
                header, run = "", 0
            elif _is_table_line(ln):
                run += 1
                if run == 1:
                    header = ln
            else:
                run = 0
        if header:
            return header
    return ""


def _content(text: str) -> set[str]:
    from finetune_studio.data.prep.qa_validate import content_tokens

    return content_tokens(text)


def _relevant_tokens(statement: Statement, pairs: Sequence[tuple[str, str]]) -> set[str]:
    """Tokens of the answers that are really ABOUT this statement.

    A bare number like "23" or "10" occurs in unrelated answers all over a chunk; counting it would call a fact covered
    that no pair states. A pair is relevant when its question+answer share enough content words with the statement.
    """
    own = _content(statement.text)
    need = max(1, min(3, round(0.4 * len(own))))
    out: set[str] = set()
    for q, a in pairs:
        if len(own & _content(f"{q} {a}")) >= need:
            out.update(canon(w) for w in _WORD.findall(a))
    return out


def is_covered(statement: Statement, pairs: Sequence[tuple[str, str]]) -> bool:
    if not statement.tokens:
        return True                     # nothing checkable in it
    have = _relevant_tokens(statement, pairs)
    hit = len(statement.tokens & have)
    n = len(statement.tokens)
    need = n if n <= STRICT_UP_TO else COVER_THRESHOLD * n
    return hit >= need


def uncovered(statements: Sequence[Statement], pairs: Sequence[tuple[str, str]]) -> list[Statement]:
    return [s for s in statements if not is_covered(s, pairs)]


def fact_coverage(statements: Sequence[Statement], pairs: Sequence[tuple[str, str]]) -> tuple[int, int]:
    """(covered, total) over statements that carry at least one distinctive token."""
    checkable = [s for s in statements if s.tokens]
    open_ = {s.idx for s in uncovered(checkable, pairs)}
    return len(checkable) - len(open_), len(checkable)


def document_title(first_chunk: str, filename: str) -> str:
    """A title for question scoping: the document's own opening line(s) when they read like a title, else a cleaned filename.

    PDFs wrap a long title over two lines ("... — Employee" / "Handbook, Version 4 (Extract)"): a short unfinished first line
    is joined with the short line after it.
    """
    from finetune_studio.data.prep.coverage_question import title_from_filename

    def looks_like_title(line: str) -> bool:
        return (6 <= len(line) <= 110 and not line.startswith("===") and " | " not in line and not line.endswith((".", ",", ";"))
                and not line.lower().startswith(("from:", "to:", "date:", "subject:", "[", "{", "<")))

    lines = [ln.strip().lstrip("#").strip() for ln in first_chunk.splitlines() if ln.strip()]
    if not lines or not looks_like_title(lines[0]):
        return title_from_filename(filename) or filename
    title = lines[0]
    short_follow_up = len(lines) > 1 and len(lines[1]) <= 60 and not lines[1].endswith((".", ";")) and len(title) + len(lines[1]) <= 110
    unfinished = title.endswith(("—", "-", "–")) or title.split()[-1][:1].isupper() or (title[-1].isalnum() and lines[1][:1].isupper() if len(lines) > 1 else False)
    return f"{title} {lines[1]}" if short_follow_up and unfinished else title


MAX_PAIRS_PER_FACT_SET = 2


def drop_redundant(new: list[dict[str, str]], kept: Sequence[dict[str, str]]) -> list[dict[str, str]]:
    """Keep at most ``MAX_PAIRS_PER_FACT_SET`` pairs that state the same set of distinctive tokens (paraphrases are useful,
    a fourth rewording of "where does cargo go bad" only costs review time)."""
    pool = [distinctive_tokens(p["a"]) for p in kept]
    out: list[dict[str, str]] = []
    for pair in new:
        toks = distinctive_tokens(pair["a"])
        if toks and sum(1 for k in pool if toks <= k) >= MAX_PAIRS_PER_FACT_SET:
            continue
        out.append(pair)
        pool.append(toks)
    return out


# ── prompts ────────────────────────────────────────────────────────────────────

EXHAUSTIVE_SYSTEM = """You turn company documents into question-and-answer training pairs. The goal is COMPLETENESS: a model trained on \
your pairs must be able to answer any question about this passage, so no fact may be left out.

Output ONLY a JSON array of {"q": "...", "a": "..."} objects. No markdown fences, no commentary, no thinking blocks. Start with [.

Rules:
- Write one pair per distinct fact: every number, amount, date, time, threshold, name, code, identifier, condition, deadline, contact, \
rule and every table row or list item. Do not summarise. Do not skip details because they look minor or boring. Several facts in one \
sentence become several pairs.
- Every question must stand alone: name the subject (document, system, policy, product, person, customer, row key) instead of "it", \
"this" or "the passage". Use the document and section given to you to make questions specific.
- Every answer is a complete sentence that states the exact values as written in the passage (keep numbers, units, codes and names \
verbatim). Never add information that is not in the passage.
- Never ask the same fact twice in different words.
- For chat logs, e-mails and minutes: state the fact itself (what was decided, measured, promised, when, by whom when that matters). Never write "What did <person> say at <time>?" questions or answers that just quote a message.
- For a table row, ask about the row by its key (code, name, id) and answer with the column names and their values.
- Skip only pure boilerplate that states no fact."""

EXHAUSTIVE_USER = """Document: {title}
Section: {section}

Passage:
\"\"\"
{chunk}
\"\"\"

Write the question-answer pairs for EVERY fact in the passage. Respond with ONLY the JSON array."""

GAP_USER = """Document: {title}
Section: {section}

Passage:
\"\"\"
{chunk}
\"\"\"

These statements from the passage are NOT covered by any question yet:
{numbered}

Write question-answer pairs that cover each of them (several pairs when a statement holds several facts). Questions must name \
their subject and stand alone; answers must state the exact values as written. Respond with ONLY the JSON array."""


_SEPARATOR_ROW = re.compile(r"^\s*:?-{2,}:?(\s*\|\s*:?-{2,}:?)*\s*$")


def keyed_chunk(chunk: str, carried_header: str = "") -> str:
    """Table rows rewritten as ``column: value | column: value`` so the model names columns correctly.

    Bare ``a | b | c`` rows made the model call a close date "the date" and glue the next_step column into the notes column.
    Prose and markers pass through; a chunk that starts mid-table uses the header carried over from the earlier chunk.
    """
    out: list[str] = []
    header: list[str] = [h.strip() for h in carried_header.split(" | ")] if carried_header else []
    prev_table = False
    for raw in chunk.splitlines():
        line = raw.strip()
        if _SEPARATOR_ROW.match(line) and " | " in line or line.strip("-| ") == "" and "|" in line:
            continue
        if _is_table_line(line):
            cells = [c.strip() for c in line.split(" | ")]
            if not prev_table and not (carried_header and not out):
                header = cells                      # a new table block: this line names the columns
                out.append(line)
            else:
                out.append(" | ".join(f"{h or f'column {i + 1}'}: {c}" for i, (h, c) in enumerate(zip(header, cells, strict=False)) if c))
            prev_table = True
            continue
        prev_table = False
        if line.startswith("==="):
            header = []
        out.append(raw.rstrip())
    return "\n".join(out)


def _with_header(chunk: str, carried_header: str) -> str:
    """The prompt form of a chunk: keyed table rows, plus the column names when the chunk starts mid-table."""
    body = keyed_chunk(chunk, carried_header)
    if not carried_header:
        return body
    return f"[Table columns, continued from earlier in the document: {carried_header}]\n{body}"


def build_exhaustive_messages(chunk: str, title: str, section: str, carried_header: str = "") -> list[dict[str, str]]:
    return [{"role": "system", "content": EXHAUSTIVE_SYSTEM},
            {"role": "user", "content": EXHAUSTIVE_USER.format(chunk=_with_header(chunk, carried_header)[:7000], title=title,
                                                               section=section or "(none)")}]


def build_gap_messages(chunk: str, title: str, section: str, missing: Sequence[Statement],
                       carried_header: str = "") -> list[dict[str, str]]:
    numbered = "\n".join(f"{i}. {s.prompt_text}" for i, s in enumerate(missing, 1))
    return [{"role": "system", "content": EXHAUSTIVE_SYSTEM},
            {"role": "user", "content": GAP_USER.format(chunk=_with_header(chunk, carried_header)[:7000], title=title,
                                                         section=section or "(none)", numbered=numbered)}]


# ── deterministic last resort ──────────────────────────────────────────────────

def extractive_pair(statement: Statement, title: str, section: str) -> dict[str, str] | None:
    """A pair quoted from the statement itself. Tables keep their column names; prose gets a scoped, honest question."""
    text = statement.text.strip()
    if statement.kind == "table" and statement.header:
        heads = [h.strip() for h in statement.header.split(" | ")]
        cells = [c.strip() for c in text.split(" | ")]
        if len(cells) >= 2 and cells[0]:
            pairs = [f"{h or f'column {i + 1}'}: {c}" for i, (h, c) in enumerate(zip(heads, cells, strict=False)) if c]
            return {"q": f"In {title}, what are the details of the row \"{cells[0]}\"?", "a": f"{cells[0]} — " + "; ".join(pairs[1:] or pairs)}
    if len(text) < 20:
        return None
    where = f"{title}, {section}" if section and section.lower() not in title.lower() else title
    lead = " ".join(text.split()[:8]).rstrip(",;:")
    return {"q": f"What does {where} state in the passage that begins \"{lead}\"?", "a": text}


@dataclass
class ChunkOutcome:
    pairs: list[tuple[dict[str, str], str]] = field(default_factory=list)   # (pair, origin)
    statements: int = 0
    covered: int = 0
    rounds: int = 0
    extractive: int = 0


def mine_chunk(
    chunk: str,
    *,
    chat: Callable[[list[dict[str, str]]], str],
    parse: Callable[[str], list[dict[str, str]]],
    accept: Callable[[list[dict[str, str]], str], list[dict[str, str]]],
    title: str,
    section: str,
    carried_header: str = "",
) -> ChunkOutcome:
    """Run the exhaustive loop for one chunk. ``chat`` is one model call, ``parse`` turns its reply into pairs and
    ``accept`` applies the validation gate (and dedup) to a batch, returning the accepted pairs."""
    out = ChunkOutcome()
    statements = split_statements(chunk, carried_header)
    checkable = [s for s in statements if s.tokens]
    out.statements = len(checkable)

    def run(messages: list[dict[str, str]], origin: str) -> None:
        try:
            reply = chat(messages)
        except Exception:  # noqa: BLE001 — a failed call is a gap the later rounds / extractive fallback close
            return
        accepted = drop_redundant(accept(parse(reply), chunk), [p for p, _ in out.pairs])
        for pair in accepted:
            out.pairs.append((pair, origin))

    run(build_exhaustive_messages(chunk, title, section, carried_header), "model")
    out.rounds = 1
    for _ in range(MAX_GAP_ROUNDS):
        missing = uncovered(checkable, [(p["q"], p["a"]) for p, _ in out.pairs])
        if not missing:
            break
        run(build_gap_messages(chunk, title, section, missing, carried_header), "model_gap")
        out.rounds += 1
    missing = uncovered(checkable, [(p["q"], p["a"]) for p, _ in out.pairs])
    for s in missing:
        pair = extractive_pair(s, title, section)
        if pair:
            kept = accept([pair], chunk)
            if kept:
                out.pairs.append((kept[0], "extractive_gap"))
                out.extractive += 1
    out.covered = out.statements - len(uncovered(checkable, [(p["q"], p["a"]) for p, _ in out.pairs]))
    return out
