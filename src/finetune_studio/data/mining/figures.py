"""Mining stage 0: the Vision reader reads a file's figures before its inventory is taken.

The parser left each picture as a ``[Figure N · where · read by OCR, not checked]`` block (``data/figures.py``). OCR keeps the
numbers of a chart but not which label they belong to (the annual report's revenue chart came out as "84.3 68.8 56.9 52.4
97.6 / 2019 2020 2021 2022 2023"), so mining could not pair them. When a Vision reader is configured, each OCR block's body
is replaced by the model's reading, ``parsed.txt`` and the chunks are rewritten, and only then is the file inventoried.

Runs only on a file nobody has mined yet: re-chunking moves every chunk index, and items and pairs are keyed by it (Reset
the file to read its figures afterwards).
"""
from __future__ import annotations

import logging
import re
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any

from finetune_studio.data import figures as figs

if TYPE_CHECKING:
    from finetune_studio.knowledge.sources import Source

log = logging.getLogger(__name__)

OCR_MARK = figs.OCR_MARK
HELD_BACK = "(a picture read only by OCR: held back from mining, because OCR loses which label each number belongs to. Set " \
            "a Vision reader in Settings, then Reset and Mine this file.)"


@dataclass
class FigureResult:
    read: int = 0
    failed: list[str] = field(default_factory=list)
    skipped: str = ""      # why nothing was read (no reader, already mined, ...)
    changed: bool = False


def pending_blocks(text: str, kind: str | None = None) -> list[tuple[int, str]]:
    """(figure number, where) of every block still holding only OCR text (of one ``kind`` when given)."""
    out = []
    for m in figs.BLOCK_RE.finditer(text):
        where, k, read_by = figs.header_parts(m.group(2))
        if read_by in figs.UNCHECKED and (kind is None or k == kind):
            out.append((int(m.group(1)), where))
    return out


def ocr_pictures(full_text: str) -> set[int]:
    """Numbers of the PICTURE blocks that hold only OCR text."""
    out = set()
    for m in figs.BLOCK_RE.finditer(full_text):
        _where, kind, read_by = figs.header_parts(m.group(2))
        if kind == figs.PICTURE and read_by in figs.UNCHECKED:
            out.add(int(m.group(1)))
    return out


_OPEN = re.compile(r"\[Figure (\d+) · [^\]]*\]\n")
_CLOSE = re.compile(r"\n?\[End of figure (\d+)\]")


def hold_back_pictures(chunk: str, pictures: set[int]) -> tuple[str, int]:
    """``chunk`` with the OCR body of the given picture figures replaced by a note (scanned pages keep theirs); how many.

    Works on a stored chunk, so a block cut by a chunk boundary is handled too: a body that runs past the chunk end, or
    one that started in the previous chunk. Without a Vision reader, mining a chart's OCR text paired values with the
    wrong labels (29 of 83 chart pairs)."""
    if not pictures:
        return chunk, 0
    out: list[str] = []
    pos, held = 0, 0
    # a body continuing from the previous chunk: everything up to its end marker
    first_close = _CLOSE.search(chunk)
    first_open = _OPEN.search(chunk)
    if first_close and (first_open is None or first_close.start() < first_open.start()) \
            and int(first_close.group(1)) in pictures:
        out.append(f"{HELD_BACK}\n[End of figure {first_close.group(1)}]")
        pos, held = first_close.end(), 1
    while (m := _OPEN.search(chunk, pos)) is not None:
        n = int(m.group(1))
        end = _CLOSE.search(chunk, m.end())
        stop = end.end() if end else len(chunk)
        out.append(chunk[pos:m.start()])
        if n in pictures:
            out.append(f"{m.group(0)}{HELD_BACK}" + (f"\n[End of figure {n}]" if end else ""))
            held += 1
        else:
            out.append(chunk[m.start():stop])
        pos = stop
    out.append(chunk[pos:])
    return "".join(out), held


_HEADER = re.compile(r"\[Figure (\d+) · ([^\]·]*?) · [^\]]*\]")
_END = re.compile(r"\n?\[End of figure \d+\]")


def neutral_markers(chunk: str) -> str:
    """Figure headers as the helper should see them: ``[Figure 2, page 3]``. The full header (kind, which model read it)
    is bookkeeping; shown to the helper it wrote pairs like "How was Figure 1 read? -> by Gemma4-12B...gguf"."""
    chunk = chunk.replace(HELD_BACK, "(picture not read)")   # the note is for people, not a fact to mine
    def head(m: re.Match[str]) -> str:
        where = m.group(2).strip().removeprefix(figs.RENDERED).strip()          # "rendered page 1" -> "page 1"
        where = f"page {where}" if where.isdigit() else where
        return f"[Figure {m.group(1)}, {where}]"

    return _END.sub("", _HEADER.sub(head, chunk))


def replace_block(text: str, n: int, new_block: str) -> str:
    """``text`` with figure ``n``'s block replaced (the first match only; numbers are unique per file)."""
    for m in figs.BLOCK_RE.finditer(text):
        if int(m.group(1)) == n:
            return text[:m.start()] + new_block + text[m.end():]
    return text


def read_file_figures(pid: str, source: Source, reader: Any,
                      progress: Callable[[str], None] = lambda _m: None) -> FigureResult:
    """Read the file's OCR figure blocks with ``reader`` (``models.vision.Reader``) and rewrite its parse."""
    from finetune_studio.knowledge import sources

    res = FigureResult()
    title = source.filename
    text = sources.parsed_text(pid, source)
    if not text:
        res.skipped = "no parsed text"
        return res
    todo = pending_blocks(text)
    if not todo:
        return res
    raw = Path(source.raw_path)
    if not raw.is_file():
        res.skipped = "the original file is gone: figures cannot be re-read"
        return res
    if any(where.startswith(figs.RENDERED) for _n, where in todo):
        # read through LibreOffice at parse time (no parser, or its text was thin): render the original again
        from finetune_studio.data.render import RenderError
        try:
            found = {f.n: f for f in figs.rendered(raw).figures}
        except RenderError as e:
            res.skipped = f"the file could not be rendered again: {e}"
            return res
    else:
        found = {f.n: f for f in figs.extract(raw)}
    for i, (n, where) in enumerate(todo, 1):
        fig = found.get(n)
        if fig is None or fig.where != where:   # the parse and the extractor disagree: never put a reading in the wrong place
            res.failed.append(f"figure {n} ({where}): not found in the original file")
            continue
        progress(f"Reading figure {i}/{len(todo)} ({where}) with {reader.label}…")
        try:
            reading = reader.read(fig.data, title, where)
        except Exception as e:  # noqa: BLE001 - one unreadable figure keeps its OCR text; the job reports it
            res.failed.append(f"figure {n} ({where}): {e}")
            continue
        if not reading:
            res.failed.append(f"figure {n} ({where}): the vision model returned nothing")
            continue
        text = replace_block(text, n, figs.block(fig, reading, reader.label))
        res.read += 1
    if res.read:
        sources.set_text(pid, source.id, text, parser=source.parser.removesuffix("+vision") + "+vision")
        res.changed = True
    return res
