"""Knowledge check rules about versions inside and between documents: amendments, older workbook sheets, year-named
files and drafts.

Tarrowfield (2026-11-12) showed what name/version rules miss: an "Amendment No. 1" changes clauses of a master agreement
without being a newer copy of it (both stay valid, so it is neither a duplicate nor out of date), a workbook keeps last
year's sheet next to this year's (``Bands 2025`` / ``Bands 2024``), ``salary_bands_2024.xlsx`` next to
``salary_bands_2025.xlsx`` carries no ``v`` number, and a lone ``…_DRAFT.docx`` with nothing newer is a draft, not an
old version. Deterministic, like the rest of ``knowledge.lint``: names, the first part of each text, references.
"""
from __future__ import annotations

import re
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any

HEAD_CHARS = 3000
_EMAIL = re.compile(r"\.(eml|msg|mbox)$", re.IGNORECASE)
_AMEND_NAME = re.compile(r"(?:^|[\W_])(?:amendment|variation|addendum)[\W_]*(?:no\.?|number)?[\W_]*(\d{1,3})(?:[\W_]|$)",
                         re.IGNORECASE)
_AMEND_TITLE = re.compile(r"\b(?:amendment|variation(?:\s+agreement)?|addendum)\s+(?:no\.?|number|#)\s*(\d{1,3})\b",
                          re.IGNORECASE)
_CHANGE_WORDS = re.compile(r"\b(supersedes|replaces\s+clause|with\s+effect\s+from|(?:is|are)\s+(?:hereby\s+)?"
                           r"(?:amended|deleted\s+and\s+replaced|varied))\b", re.IGNORECASE)
_REF = re.compile(r"\b[A-Z]{2,6}(?:-[A-Z0-9]{2,8}){2,4}\b")         # TWL-MSA-2025-014, LDC-TWI-2024-03
_NAME_TOKEN = re.compile(r"[a-z]{4,}")
_GENERIC = frozenset(["amendment", "variation", "addendum", "signed", "scan", "scanned", "effective", "agreement",
                      "master", "lease", "contract", "final", "draft", "copy", "version", "order", "form",
                      "services", "service", "schedule", "annex", "appendix", "mutual", "notes", "summary", "report",
                      "letter", "policy", "terms"])
_SHEET = re.compile(r"^=== Sheet: (?P<name>.+?) \(\d+ rows\) ===$", re.MULTILINE)
_YEAR = re.compile(r"(?<!\d)((?:19|20)\d{2})(?!\d)")
_BARE_YEAR = re.compile(r"(?<![^\W_])((?:19|20)\d{2})(?![^\W_])")      # a year alone, not part of a date 2025-09-30
_OLD_SHEET_TEXT = re.compile(r"\b(superseded|previous|old|archived?|obsolete|replaced)\b", re.IGNORECASE)
_DRAFT_TOKENS = re.compile(r"(?:^|[\W_])(?:draft|v(?:er(?:sion)?)?[\W_]?\d{1,3}|final|wip)(?=[\W_]|$)", re.IGNORECASE)


@dataclass
class Amended:
    base: Any
    amendment: Any
    number: str
    evidence: list[str] = field(default_factory=list)


def _norm(s: str) -> str:
    return re.sub(r"[\W_]+", "", s).lower()


def is_amendment(source: Any, text: str) -> str:
    """The amendment's number ("1") when the file is an amendment by its name or its title, else "". E-mails about
    an amendment are not one."""
    if _EMAIL.search(source.filename):
        return ""
    m = _AMEND_NAME.search(source.filename) or _AMEND_TITLE.search(text[:400])
    return m.group(1) if m else ""


def _tokens(name: str) -> set[str]:
    return {t for t in _NAME_TOKEN.findall(re.sub(r"\.[^.]+$", "", name).lower()) if t not in _GENERIC}


def amendments(sources: list[Any], texts: dict[str, str]) -> list[Amended]:
    """Each amendment with the document it amends: the one whose name or opening names a reference the amendment
    cites (``TWL-MSA-2025-014``), else the one sharing the most distinctive name words (``Brannock``)."""
    out: list[Amended] = []
    numbered = {s.id: is_amendment(s, texts.get(s.id, "")) for s in sources}
    bases = [s for s in sources if not numbered[s.id] and not _EMAIL.search(s.filename)]
    for a in sources:
        number = numbered[a.id]
        if not number:
            continue
        head = texts.get(a.id, "")[:HEAD_CHARS]
        refs = {r for r in _REF.findall(head) if _norm(r) not in _norm(a.filename)}
        refs = {r for r in refs if not any(r != o and r in o for o in refs)}     # "ML-2022-087" inside "TWL-ML-2022-087"
        best, best_score, why = None, 0.0, ""
        for b in bases:
            b_head = texts.get(b.id, "")[:HEAD_CHARS]
            hit = sorted(r for r in refs if _norm(r) in _norm(b.filename) or r in b_head)
            shared = _tokens(a.filename) & _tokens(b.filename)
            score = 10.0 * len(hit) + len(shared)
            if score > best_score:
                best, best_score = b, score
                why = (f"it cites {', '.join(hit)}" if hit else f"the names share {', '.join(sorted(shared))}")
        if best is None:
            continue
        evidence = [f"Amendment No. {number}: {why}"]
        changes = sorted({m.group(1).lower() for m in _CHANGE_WORDS.finditer(texts.get(a.id, ""))})
        if changes:
            evidence.append("it says " + ", ".join(f'"{c}"' for c in changes))
        out.append(Amended(best, a, number, evidence))
    return out


def _family(name: str) -> str:
    return _norm(_YEAR.sub("", name))


def older_sheets(text: str) -> list[tuple[str, str]]:
    """``(older sheet, newer sheet)`` of one workbook's parse: sheets named alike but for a year (``Bands 2024`` next to
    ``Bands 2025``), and a sheet whose first line calls itself superseded or previous (paired with the sheet before it
    when no year says otherwise)."""
    marks = list(_SHEET.finditer(text))
    sheets = []
    for i, m in enumerate(marks):
        body = text[m.end():marks[i + 1].start() if i + 1 < len(marks) else len(text)]
        first = next((ln for ln in body.splitlines() if ln.strip(" |")), "")
        sheets.append((m.group("name"), first))
    out: list[tuple[str, str]] = []
    by_family: dict[str, list[tuple[int, str]]] = defaultdict(list)
    for name, _first in sheets:
        years = _YEAR.findall(name)
        if len(years) == 1:
            by_family[_family(name)].append((int(years[0]), name))
    for family in by_family.values():
        if len({y for y, _ in family}) > 1:
            newest = max(family)
            out.extend((name, newest[1]) for y, name in family if y < newest[0])
    named = {old for old, _ in out}
    for i, (name, first) in enumerate(sheets):
        if name not in named and _OLD_SHEET_TEXT.search(first) and len(sheets) > 1:
            newer = sheets[i - 1][0] if i > 0 else sheets[1][0]
            out.append((name, newer))
    return out


def year_key(name: str) -> tuple[str, int] | None:
    """(family, year) of a file named with a year alone (``salary_bands_2024.xlsx``); None for a full date
    (``Board_minutes_2025-05-22``: a series, not versions) or no year."""
    stem = re.sub(r"\.[^.]+$", "", name)
    if re.search(r"(?:19|20)\d{2}[-_.]\d{2}", stem):
        return None
    years = _BARE_YEAR.findall(stem)
    if len(years) != 1:
        return None
    return _norm(_BARE_YEAR.sub("", stem)), int(years[0])


def draft_family(name: str) -> str:
    """A file name without its draft/version/final words and year: the name its final version would share."""
    stem = re.sub(r"\.[^.]+$", "", name)
    return _norm(_BARE_YEAR.sub("", _DRAFT_TOKENS.sub("_", stem)))
