"""App knowledge base: one markdown file per page/feature, loaded and cached.

WHAT THIS FILE DOES
===================
Reads ``guide/kb/*.md`` (version-controlled with the code) into ``KbEntry``
objects. Each file has a tiny ``key: value`` front matter (``id``, ``title``,
``page``, ``keywords``; optional ``controls`` = page that owns the listed controls) and ``## Section`` bodies (Purpose, Workflow, Key
controls, What to check, Common mistakes). The element ids named in a Key
controls section are verified against the real templates by
``tests/test_guide_kb.py``, so the KB cannot drift from the UI unnoticed.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path

from finetune_studio.guide.registry import PAGES

KB_DIR = Path(__file__).parent / "kb"
SECTION_ORDER = ("Purpose", "Workflow", "Key controls", "What to check", "Common mistakes")
_CONTROL_RE = re.compile(r"^- `([^`]+)`", re.MULTILINE)


@dataclass(frozen=True)
class KbEntry:
    id: str
    title: str
    page: str                       # key into registry.PAGES
    keywords: tuple[str, ...]
    sections: dict[str, str] = field(default_factory=dict)
    source: str = ""
    controls_page: str = ""         # page whose template holds the Key controls (default: ``page``)

    @property
    def controls(self) -> list[str]:
        """Selectors named in the Key controls section (``#id`` or ``[name="x"]``)."""
        return _CONTROL_RE.findall(self.sections.get("Key controls", ""))

    @property
    def route(self) -> str:
        return PAGES[self.page].url if self.page in PAGES else ""

    def render(self, max_chars: int = 3000) -> str:
        parts = [f"{self.title} (page: {self.route or 'n/a'})"]
        for name in SECTION_ORDER:
            if name in self.sections:
                parts.append(f"## {name}\n{self.sections[name]}")
        return "\n\n".join(parts)[:max_chars]


def _parse(path: Path) -> KbEntry:
    raw = path.read_text(encoding="utf-8")
    match = re.match(r"---\n(.*?)\n---\n(.*)", raw, re.DOTALL)
    if not match:
        raise ValueError(f"{path.name}: missing front matter")
    meta: dict[str, str] = {}
    for line in match.group(1).splitlines():
        key, _, value = line.partition(":")
        meta[key.strip()] = value.strip()
    for required in ("id", "title", "page", "keywords"):
        if not meta.get(required):
            raise ValueError(f"{path.name}: front matter lacks {required!r}")
    sections: dict[str, str] = {}
    for chunk in re.split(r"^## ", match.group(2), flags=re.MULTILINE)[1:]:
        name, _, body = chunk.partition("\n")
        sections[name.strip()] = body.strip()
    return KbEntry(
        id=meta["id"], title=meta["title"], page=meta["page"],
        keywords=tuple(meta["keywords"].split()), sections=sections, source=path.name,
        controls_page=meta.get("controls", meta["page"]),
    )


@lru_cache(maxsize=1)
def load_kb() -> dict[str, KbEntry]:
    """All entries by id, in file order."""
    entries: dict[str, KbEntry] = {}
    for path in sorted(KB_DIR.glob("*.md")):
        entry = _parse(path)
        if entry.id in entries:
            raise ValueError(f"duplicate KB id {entry.id!r} ({path.name})")
        entries[entry.id] = entry
    return entries


def table_of_contents() -> list[dict[str, str]]:
    return [{"id": e.id, "title": e.title, "page": e.route} for e in load_kb().values()]
