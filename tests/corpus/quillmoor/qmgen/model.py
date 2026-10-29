"""The document model shared by the world builder (``docs.py``) and the format writers (``render.py``).

A ``Doc`` is format-neutral: a title, metadata and a list of ``Block``s. The builder decides the content and records,
for every fact it writes, the exact string it put on the page (``Doc.facts``); the writers only lay the blocks out in
their format and never change a block's text. That is what lets the benchmark ask "did this exact value survive
parsing / reach the index / come back in a search hit" without guessing at number or date formats.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class Block:
    """One layout element.

    kind:
      heading  -- ``text``, ``level`` 1..3
      para     -- ``text`` (one paragraph)
      bullets  -- ``items`` (one string each)
      table    -- ``columns`` + ``rows`` (all strings), optional ``caption`` (also the sheet name in spreadsheets)
      kv       -- ``pairs`` of (label, value): a form / spec sheet / nameplate
      quote    -- ``text``: a quoted earlier message in an email or chat
      note     -- ``text``: a footnote, side remark, footer line
      pagebreak -- a new page (pdf/docx/odt) or a new slide (pptx/odp); ignored by flat formats
    """

    kind: str
    text: str = ""
    level: int = 1
    items: list[str] = field(default_factory=list)
    columns: list[str] = field(default_factory=list)
    rows: list[list[str]] = field(default_factory=list)
    pairs: list[tuple[str, str]] = field(default_factory=list)
    caption: str = ""


@dataclass
class FactMark:
    """A fact this document states, and the literal string it wrote for the value."""

    fact_id: str
    written: str            # exact substring present in the document text
    superseded: bool = False  # the doc states an outdated value (an older version of the policy, a corrected email)


@dataclass
class Doc:
    doc_id: str                    # stable id, also the file stem
    kind: str                      # site_profile, asset_register, incident_report, ... (docs.py)
    ext: str                       # target extension with the dot, e.g. ".docx"
    title: str
    blocks: list[Block] = field(default_factory=list)
    # Format-specific inputs (each optional):
    raw: str | None = None         # exact file text for code / config / log / chat / json / yaml / xml / jsonl / csv
    meta: dict[str, Any] = field(default_factory=dict)
    # meta keys used by writers:
    #   author, date (ISO), company            -- document properties / headers / footers
    #   email: {"from", "to", "cc", "subject", "date" (RFC 2822), "message_id", "in_reply_to"}
    #   scan: True                              -- a .pdf rendered as page images only (OCR path)
    #   image_style: "nameplate" | "notice" | "whiteboard"   -- for image extensions
    #   header / footer: str                    -- running header/footer text (docx/pdf/odt)
    facts: list[FactMark] = field(default_factory=list)
    subdir: str = ""               # folder inside the corpus (e.g. "ops/registers")

    @property
    def filename(self) -> str:
        return f"{self.doc_id}{self.ext}"

    @property
    def relpath(self) -> str:
        return f"{self.subdir}/{self.filename}" if self.subdir else self.filename

    def plain_text(self) -> str:
        """The block text as one string (what a perfect parser would extract), for self-checks."""
        if self.raw is not None:
            return self.raw
        out: list[str] = [self.title]
        for b in self.blocks:
            if b.kind in ("heading", "para", "quote", "note"):
                out.append(b.text)
            elif b.kind == "bullets":
                out.extend(b.items)
            elif b.kind == "table":
                if b.caption:
                    out.append(b.caption)
                out.append(" | ".join(b.columns))
                out.extend(" | ".join(r) for r in b.rows)
            elif b.kind == "kv":
                out.extend(f"{k}: {v}" for k, v in b.pairs)
        return "\n".join(out)
