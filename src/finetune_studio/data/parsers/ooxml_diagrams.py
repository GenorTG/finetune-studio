"""SmartArt in Office files (docx, pptx) as an indented outline.

SmartArt keeps its text in a data part (``word/diagrams/data1.xml``, ``ppt/diagrams/data1.xml``): points (``dgm:pt``) with
their text, and connections (``dgm:cxn``, parent -> child, ``srcOrd`` = order among siblings). The text parsers skipped it,
so an org chart or a process diagram reached mining as nothing; this reads it exactly (no rendering, no vision)::

    [Diagram]
    - Duty Ops
      - Quality on-call
      - Fleet
"""
from __future__ import annotations

from xml.etree import ElementTree as ET

_DGM = "http://schemas.openxmlformats.org/drawingml/2006/diagram"
_A = "http://schemas.openxmlformats.org/drawingml/2006/main"
REL_ATTR = "{http://schemas.openxmlformats.org/officeDocument/2006/relationships}dm"   # on dgm:relIds: the data part
REL_IDS_TAG = f"{{{_DGM}}}relIds"
_SKIP_TYPES = {"parTrans", "sibTrans", "pres"}


def diagram_outline(xml: bytes) -> str:
    """One SmartArt data part as ``[Diagram]`` + an indented ``- text`` outline; ``""`` when it holds no text."""
    try:
        root = ET.fromstring(xml)
    except ET.ParseError:
        return ""
    text: dict[str, str] = {}
    doc_ids: list[str] = []
    for pt in root.iter(f"{{{_DGM}}}pt"):
        mid, kind = pt.get("modelId", ""), pt.get("type", "node")
        if kind == "doc":
            doc_ids.append(mid)
        if kind in _SKIP_TYPES or kind == "doc":
            continue
        t = " ".join("".join(x.text or "" for x in pt.iter(f"{{{_A}}}t")).split())
        if t:
            text[mid] = t
    children: dict[str, list[tuple[int, str]]] = {}
    has_parent: set[str] = set()
    for c in root.iter(f"{{{_DGM}}}cxn"):
        if c.get("type", "parOf") != "parOf":
            continue
        src, dst = c.get("srcId", ""), c.get("destId", "")
        children.setdefault(src, []).append((int(c.get("srcOrd", "0") or 0), dst))
        has_parent.add(dst)
    lines: list[str] = []
    seen: set[str] = set()

    def walk(mid: str, depth: int) -> None:
        if mid in seen:
            return
        seen.add(mid)
        nxt = depth
        if mid in text:
            lines.append(f"{'  ' * depth}- {text[mid]}")
            nxt = depth + 1
        for _ord, child in sorted(children.get(mid, [])):
            walk(child, nxt)

    for d in doc_ids:
        walk(d, 0)
    for mid in text:                      # points no connection reaches still carry text
        if mid not in seen and mid not in has_parent:
            walk(mid, 0)
    return "[Diagram]\n" + "\n".join(lines) if lines else ""
