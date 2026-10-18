"""Rule-based data points: cut a chunk into statements and decide whether pairs state them.

``split_statements`` cuts a chunk into statements (a table row, a bullet, a sentence), each with its *distinctive tokens*:
numbers, dates, codes, capitalised names, the things a reader could be asked about. A statement is covered when its tokens
appear in the answers of the pairs that are really about it (``is_covered``). This is the rule half of a file's inventory; the
agent half (``inventory``) adds facts stated in words.
"""
from __future__ import annotations

import re
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import PurePath

from finetune_studio.data.prep.qa_validate import content_tokens
from finetune_studio.data.prep.tokens import (
    _CODE,
    _NUMBERISH,
    _WORD,
    canon,
    distinctive_tokens,
)

# A statement counts as covered when ALL its distinctive tokens appear in the relevant answers (a phone number or a
# certificate id is exactly the token a "85 % is enough" rule drops); long statements may miss up to 10 % of theirs.
COVER_THRESHOLD = 0.9
STRICT_UP_TO = 8
# Statements longer than this are cut at ';' / ' — ' so one statement is one or two facts, not a paragraph.
MAX_STATEMENT_CHARS = 420
MIN_SENTENCE_CHARS = 25
# At most this many pairs may state the same set of distinctive tokens (a paraphrase helps, a fourth one costs review time).
MAX_PAIRS_PER_FACT_SET = 2


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


# ── sentences ─────────────────────────────────────────────────────────────────────────────────────────────────────────

_SENT_RE = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9])")
# A period after these does not end a sentence ("...given by Dr. Maren Voss." must not become "...given by Dr.").
_ABBREVIATIONS = frozenset({
    "dr", "mr", "mrs", "ms", "prof", "sr", "jr", "st", "vs", "etc", "no", "fig", "inc", "ltd", "co", "cf",
    "approx", "dept", "mt", "gen", "col", "lt", "sgt", "capt", "rev", "hon", "e.g", "i.e",
})


def _ends_with_abbreviation(piece: str) -> bool:
    last = piece.rstrip().rsplit(" ", 1)[-1].rstrip(".").lower()
    return last in _ABBREVIATIONS or (len(last) == 1 and last.isalpha() and piece.rstrip().endswith("."))   # "J. Smith"


def split_sentences(text: str) -> list[str]:
    """Deterministic, abbreviation-aware sentence split; pieces shorter than ``MIN_SENTENCE_CHARS`` are dropped."""
    text = re.sub(r"\s+", " ", text or "").strip()
    if not text:
        return []
    merged: list[str] = []
    for piece in _SENT_RE.split(text):
        if merged and _ends_with_abbreviation(merged[-1]):
            merged[-1] = f"{merged[-1]} {piece}"
        else:
            merged.append(piece)
    return [s.strip() for s in merged if len(s.strip()) >= MIN_SENTENCE_CHARS]


# ── statements ────────────────────────────────────────────────────────────────────────────────────────────────────────

def _is_table_line(line: str) -> bool:
    return line.count(" | ") >= 1


_LIST_START = re.compile(r"^([-*•]|\d+[.)])\s+")


def _join_wrapped(lines: list[str]) -> list[str]:
    """Re-join prose a PDF/RTF parser wrapped mid-sentence: a line that starts in lower case continues an unfinished one.

    ("... carried over into the following" / "year, but ..." were two statements and the number landed in neither; a line may
    also continue with a parenthesis: "... (code GM-" / "(GM-2); theft ...".)
    """
    out: list[str] = []
    for raw in lines:
        line = raw.strip()
        prev = out[-1].strip() if out else ""
        if (line and prev and (line[0].islower() or line[0] == "(") and not prev.endswith((".", "!", "?", ":", ";"))
                and not _is_table_line(line) and not _is_table_line(prev) and not prev.startswith("===")):
            out[-1] = f"{prev} {line}"
        else:
            out.append(raw)
    return out


def split_statements(chunk: str, carried_header: str = "") -> list[Statement]:
    """Cut a chunk into statements; table rows carry their header (from this chunk or ``carried_header``).

    A table's header line itself is not a statement (column names are not facts); its rows are.
    """
    statements: list[Statement] = []
    header = ""
    prev_table = False
    seen_content = False
    for raw in _join_wrapped(chunk.splitlines()):
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
                    prev_table = seen_content = True
                    continue
            prev_table = seen_content = True
            statements.append(Statement(len(statements), line, "table", distinctive_tokens(line), header))
            continue
        prev_table = False
        seen_content = True
        kind = "list" if _LIST_START.match(line) else "prose"
        pieces = [line] if kind == "list" else split_sentences(line)
        for piece in pieces:
            for part in _cut_long(piece):
                if len(part.strip()) >= 12:
                    statements.append(Statement(len(statements), part.strip(), kind, distinctive_tokens(part)))
    return statements


# "[2024-09-18 22:12] olek:" / "[22:12] olek:" — when a chat message was posted, not a fact it states.
_CHAT_STAMP = re.compile(r"^\[[^\]]{4,40}\]\s*[\w.\- ]{1,40}:\s*")
# "5.2 ", "17.4 ", "3) ", "(b) " at the start: the clause/list number, which no answer states as data.
_CLAUSE_NO = re.compile(r"^(\d+(\.\d+)+\.?|\d+[.)]|\(?[a-z0-9]{1,3}\))\s+", re.IGNORECASE)
_THOUSANDS = re.compile("(?<=\\d)[,\u00a0\u202f](?=\\d{3}(?!\\d))")   # comma, no-break / thin space; never a plain space


def join_thousands(text: str) -> str:
    """``4,000,000`` -> ``4000000`` so a large number is one token (the tokenizer split it into ``4`` + ``000``)."""
    return _THOUSANDS.sub("", text)


def hard_tokens(text: str) -> frozenset[str]:
    """The data a statement must have stated: numbers, dates, times, codes and ALLCAPS identifiers.

    Plain capitalised words ("Company", "Date", a name) are left to the helper's inventory, which judges meaning: as hard
    tokens they turned headings and chatter into data points no pair needs to state. A chat line's leading timestamp and
    author are metadata, not part of what it says.
    """
    body = join_thousands(_CLAUSE_NO.sub("", _CHAT_STAMP.sub("", text.strip())))
    out: set[str] = set()
    for w in _WORD.findall(body):
        c = canon(w)
        if len(c) < 2:
            continue
        bare = w.strip(".,;:()")
        # "RC-01", "KC-Tag": a capital or digit after the first letter; never "cut-off" or a sentence-initial "E-mail"
        code = (_CODE.fullmatch(bare) and any(ch.isupper() or ch.isdigit() for ch in bare[1:])
                and (any(ch in "-_/" for ch in bare) or any(ch.isdigit() for ch in bare)))      # "AG", "EU" are not codes
        if _NUMBERISH.search(c) or code or (w.isupper() and len(w) > 2):
            out.add(c)
    return frozenset(out)


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
    """Header row of the table chunk ``idx`` (0-based) continues: scan earlier chunks back to the last table-block start."""
    header = ""
    in_table = False
    for prev in range(max(idx, 0)):
        for raw in chunks[prev].splitlines():
            ln = raw.strip()
            if not ln:
                continue
            if ln.startswith("==="):
                header, in_table = "", False
            elif _is_table_line(ln):
                if not in_table:
                    header = ln
                in_table = True
            else:
                in_table = False
    return header if in_table else ""


# ── coverage ──────────────────────────────────────────────────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class PairTokens:
    """A pair prepared once for many statement checks."""
    pair_id: str
    content: frozenset[str]     # content words of question + answer (relevance)
    answer: frozenset[str]      # canonical words of the answer (what it states)


_NUM_WITH_UNIT = re.compile(r"^(\d[\d.]*?)(?:[a-z%°]+)$")


def pair_tokens(pair_id: str, question: str, answer: str) -> PairTokens:
    words = {canon(w) for w in _WORD.findall(join_thousands(answer))}
    words |= {m.group(1) for w in words if (m := _NUM_WITH_UNIT.match(w))}     # "57.5m" also states "57.5"
    return PairTokens(pair_id, frozenset(content_tokens(f"{question} {answer}")), frozenset(words))


def covering_pairs(text: str, tokens: frozenset[str], pairs: Sequence[PairTokens]) -> list[str]:
    """Ids of the pairs that state this statement, or ``[]`` when together they do not.

    A pair counts when it is really ABOUT the statement (shares enough content words with it: a bare "23" occurs in unrelated
    answers all over a file); the statement is covered when those pairs' answers hold all its tokens (90 % for long ones).
    """
    if not tokens:
        return []
    own = content_tokens(text)
    need_overlap = max(1, min(3, round(0.4 * len(own))))
    relevant = [p for p in pairs if len(own & p.content) >= need_overlap]
    have: set[str] = set()
    for p in relevant:
        have |= p.answer
    hit = len(tokens & have)
    need = len(tokens) if len(tokens) <= STRICT_UP_TO else COVER_THRESHOLD * len(tokens)
    if hit < need:
        return []
    return [p.pair_id for p in relevant if tokens & p.answer]


# ── document scope + prompt shape ────────────────────────────────────────────────────────────────────────────────────

_HEX_RE = re.compile(r"^[0-9a-f]{6,}$|^[0-9a-f-]{20,}$", re.IGNORECASE)


_YEAR_RE = re.compile(r"^(19|20)\d{2}$")


def title_from_filename(filename: str) -> str:
    """Readable topic from a file name; ``""`` for hashes. Years stay ("board_pack_q3_2024" is not "Board Pack Q3")."""
    stem = PurePath(filename or "").stem
    words = [w for w in re.split(r"[\s_\-.]+", stem)
             if w and (not w.isdigit() or _YEAR_RE.match(w)) and not _HEX_RE.match(w)]
    if not words:
        return ""
    title = " ".join(words)
    return title.title() if title.islower() else title


# Parser/reader markers in front of the real text: "[Figure 1 · page 1 · scanned page · read by …]", "=== Slide 1 ===".
_MARKER_LINE = re.compile(r"^(?:\[(?:figure|page|slide|sheet|image|table)\b[^\]]*\]|={2,}.*={2,})$", re.IGNORECASE)
_SUBJECT_LINE = re.compile(r"^subject:\s*(?:(?:re|fwd?|aw|wg)\s*:\s*)*(.+)$", re.IGNORECASE)
# "CAR no. KC-CAR-24-088", "Ticket #INC-2041": the record's own number, not a form or template code.
_RECORD_NO = re.compile(r"(?:\b(?i:no|nr|number|ref)\b\.?|#)\s*:?\s*([A-Z][A-Z0-9]*(?:[-/][A-Z0-9]+)+)")
_HEADING_SEPS = frozenset({"-", "–", "—", "&", "/", "·", "|"})


def _caps_heading(line: str) -> str:
    """The ALL-CAPS heading that opens a long line ("KORVANE COLD CHAIN - CORRECTIVE ACTION REQUEST - form …"), title-cased."""
    words: list[str] = []
    for tok in line.split():
        core = tok.strip("()[],;:")
        if tok in _HEADING_SEPS or (any(c.isalpha() for c in core) and core == core.upper()):
            words.append(tok)
        else:
            break
    while words and words[-1] in _HEADING_SEPS:
        words.pop()
    if sum(1 for w in words if any(c.isalpha() for c in w)) < 2 or len(" ".join(words)) < 8:
        return ""
    return " ".join(w.title() if w.isalpha() else w for w in words)


def document_title(first_chunk: str, filename: str) -> str:
    """A title for question scoping: the document's own opening line(s) when they read like a title, else the file name.

    Reader markers are skipped, an e-mail is named by its Subject, and a scanned form read as one long line by its caps
    heading plus record number: all three fell back to the file name, so 40 pairs asked about "the Scanned Corrective
    Action" instead of corrective action request KC-CAR-24-088 (walkthrough 2026-10-18)."""
    def looks_like_title(line: str) -> bool:
        return (6 <= len(line) <= 110 and not line.startswith("===") and " | " not in line and not line.endswith((".", ",", ";"))
                and not line.lower().startswith(("from:", "to:", "date:", "subject:", "[", "{", "<")))

    lines = [ln.strip().lstrip("#").strip() for ln in first_chunk.splitlines() if ln.strip()]
    while lines and _MARKER_LINE.match(lines[0]):
        lines.pop(0)
    for ln in lines[:15]:
        subject = _SUBJECT_LINE.match(ln)
        if subject and 6 <= len(subject.group(1).strip()) <= 110:
            return subject.group(1).strip()
    if lines and len(lines[0]) > 110:
        heading = _caps_heading(lines[0])
        if heading:
            record = _RECORD_NO.search(lines[0][:400])
            return f"{heading} {record.group(1)}" if record and record.group(1) not in heading else heading
    if not lines or not looks_like_title(lines[0]):
        return title_from_filename(filename) or filename
    title = lines[0]
    short_follow_up = len(lines) > 1 and len(lines[1]) <= 60 and not lines[1].endswith((".", ";")) and len(title) + len(lines[1]) <= 110
    unfinished = title.endswith(("—", "-", "–")) or title.split()[-1][:1].isupper() or (
        title[-1].isalnum() and lines[1][:1].isupper() if len(lines) > 1 else False)
    return f"{title} {lines[1]}" if short_follow_up and unfinished else title


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
    body = "\n".join(out)
    if not carried_header:
        return body
    return f"[Table columns, continued from earlier in the document: {carried_header}]\n{body}"


def drop_redundant(new: list[dict[str, str]], kept: Sequence[dict[str, str]]) -> list[dict[str, str]]:
    """Keep at most ``MAX_PAIRS_PER_FACT_SET`` pairs that state the same set of distinctive tokens."""
    pool = [distinctive_tokens(p["a"]) for p in kept]
    out: list[dict[str, str]] = []
    for pair in new:
        toks = distinctive_tokens(pair["a"])
        if toks and sum(1 for k in pool if toks <= k) >= MAX_PAIRS_PER_FACT_SET:
            continue
        out.append(pair)
        pool.append(toks)
    return out
