"""Native charts in Office files (docx, pptx, xlsx) as data tables.

A chart in an Office file keeps its numbers in ``*/charts/chartN.xml`` (the cached category labels and series values Office
draws from). The text parsers skipped charts, so a bar chart of quarterly revenue reached mining as nothing at all. This
reads that cache exactly: no OCR, no vision model, every value as Office stores it.

Output per chart, so mining sees a normal table::

    [Chart: Revenue by quarter (bar chart)]
    Category | 2023 | 2024
    Q1 | 24.1 | 26.0
"""
from __future__ import annotations

import re
import zipfile
from collections.abc import Callable
from pathlib import Path
from xml.etree import ElementTree as ET

_NS = {"c": "http://schemas.openxmlformats.org/drawingml/2006/chart",
       "a": "http://schemas.openxmlformats.org/drawingml/2006/main"}
_CHART_PART = re.compile(r"^(?:word|ppt|xl)/charts/chart\d+\.xml$")
_KIND = re.compile(r"^\{[^}]+\}(\w+?)(3D)?Chart$")


Resolver = Callable[[str], list[str]]


def _points(node: ET.Element | None, resolve: Resolver | None = None) -> list[str]:
    """Cached points of a ``c:cat``/``c:val``/``c:tx`` node, in index order (a missing index stays empty).

    Without a cache (openpyxl and other tools write only the formula) ``resolve`` reads the ``c:f`` range from the cells."""
    if node is None:
        return []
    f = node.find(".//c:f", _NS)
    if resolve is not None and f is not None and f.text and node.find(f".//{{{_NS['c']}}}pt") is None:
        return resolve(f.text.strip())
    pts: dict[int, str] = {}
    for pt in node.iter(f"{{{_NS['c']}}}pt"):
        v = pt.find("c:v", _NS)
        pts[int(pt.get("idx", len(pts)))] = (v.text or "").strip() if v is not None else ""
    if not pts:
        v = node.find(".//c:v", _NS)
        return [(v.text or "").strip()] if v is not None and v.text else []
    return [pts.get(i, "") for i in range(max(pts) + 1)]


def _title(root: ET.Element) -> str:
    t = root.find("c:chart/c:title", _NS)
    if t is None:
        return ""
    return " ".join("".join(x.text or "" for x in t.iter(f"{{{_NS['a']}}}t")).split())


def chart_table(xml: bytes, resolve: Resolver | None = None) -> str:
    """One chart part as ``[Chart: title (kind chart)]`` plus a ``Category | series...`` table; ``""`` without data."""
    try:
        root = ET.fromstring(xml)
    except ET.ParseError:
        return ""
    plot = root.find("c:chart/c:plotArea", _NS)
    if plot is None:
        return ""
    kinds: list[str] = []
    series: list[tuple[str, list[str], list[str]]] = []
    for child in plot:
        m = _KIND.match(child.tag)
        if not m:
            continue
        kinds.append(m.group(1))
        for ser in child.findall("c:ser", _NS):
            name = " ".join(_points(ser.find("c:tx", _NS), resolve)) or f"Series {len(series) + 1}"
            cats = _points(ser.find("c:cat", _NS), resolve) or _points(ser.find("c:xVal", _NS), resolve)
            vals = _points(ser.find("c:val", _NS), resolve) or _points(ser.find("c:yVal", _NS), resolve)
            if vals:
                series.append((name, cats, vals))
    if not series:
        return ""
    cats = max((s[1] for s in series), key=len)
    n = max(len(cats), *(len(s[2]) for s in series))
    head = f"[Chart: {_title(root) or 'untitled'} ({'/'.join(dict.fromkeys(kinds))} chart)]"
    rows = [" | ".join(["Category", *(s[0] for s in series)])]
    for i in range(n):
        label = cats[i] if i < len(cats) and cats[i] else str(i + 1)
        rows.append(" | ".join([label, *((s[2][i] if i < len(s[2]) else "") for s in series)]))
    return "\n".join([head, *rows])


_REF = re.compile(r"^'?(?P<sheet>.+?)'?!\$?(?P<c1>[A-Z]+)\$?(?P<r1>\d+)(?::\$?(?P<c2>[A-Z]+)\$?(?P<r2>\d+))?$")


def _col(letters: str) -> int:
    n = 0
    for ch in letters:
        n = n * 26 + ord(ch) - 64
    return n - 1


def sheet_resolver(sheets: dict[str, list[list[str]]]) -> Resolver:
    """Read ``Sheet1!$B$2:$B$4`` from parsed rows (``sheets[name][row][col]``, 0-based, as the cells were read)."""
    def resolve(ref: str) -> list[str]:
        m = _REF.match(ref)
        rows = sheets.get(m.group("sheet")) if m else None
        if not m or rows is None:
            return []
        c1, r1 = _col(m.group("c1")), int(m.group("r1")) - 1
        c2, r2 = _col(m.group("c2") or m.group("c1")), int(m.group("r2") or m.group("r1")) - 1
        return [(rows[r][c] if r < len(rows) and c < len(rows[r]) else "")
                for r in range(r1, r2 + 1) for c in range(c1, c2 + 1)]
    return resolve


def charts_in_package(path: Path, resolve: Resolver | None = None) -> list[str]:
    """Every chart of an Office file, in part order; ``[]`` for a file that is not a zip package."""
    try:
        with zipfile.ZipFile(path) as z:
            names = sorted((n for n in z.namelist() if _CHART_PART.match(n)),
                           key=lambda n: int(re.search(r"(\d+)\.xml$", n).group(1)))  # type: ignore[union-attr]
            return [t for t in (chart_table(z.read(n), resolve) for n in names) if t]
    except (zipfile.BadZipFile, OSError, KeyError):
        return []
