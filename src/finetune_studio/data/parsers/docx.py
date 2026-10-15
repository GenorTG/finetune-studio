"""DOCX (modern Word) parser."""

from __future__ import annotations

from pathlib import Path

from .. import figures as figs
from ..ocr import DEFAULT_LANGS
from ._base import cell_text, cli_run, make_result
from .ooxml_charts import chart_table
from .ooxml_diagrams import REL_ATTR, REL_IDS_TAG, diagram_outline

_CHART_TAG = "{http://schemas.openxmlformats.org/drawingml/2006/chart}chart"
_REL_ID = "{http://schemas.openxmlformats.org/officeDocument/2006/relationships}id"


def parse(path: Path) -> dict:
    try:
        from docx import Document
    except ImportError as e:
        return make_result("", {"type": "docx", "error": str(e)}, parser="docx_v1",
                           warnings=["install python-docx (pip install python-docx)"])
    from docx.table import Table
    from docx.text.paragraph import Paragraph

    doc = Document(str(path))
    paragraphs: list[str] = []
    headings = []
    tables = []
    charts = 0
    diagrams = 0
    lines: list[str] = []  # document order: a table stays under the heading/caption it was written below
    warnings: list[str] = []
    found = figs.extract(path)
    for idx, child in enumerate(doc.element.body.iterchildren()):
        here = [f for f in found if f.anchor == idx]
        if here:  # a picture is its own paragraph (often with no text): its block goes where it sits
            lines.extend(figs.ocr_block(f, DEFAULT_LANGS, warnings) for f in here)
        tag = child.tag.rsplit("}", 1)[-1]
        if tag == "p":
            p = Paragraph(child, doc)
            for ref in child.iter(_CHART_TAG):  # an inline chart: its data, where the reader sees it
                part = doc.part.related_parts.get(ref.get(_REL_ID, ""))
                if part is not None and (chart := chart_table(part.blob)):
                    lines.append(chart)
                    charts += 1
            for ref in child.iter(REL_IDS_TAG):  # SmartArt: its outline from the data part, where it sits
                part = doc.part.related_parts.get(ref.get(REL_ATTR, ""))
                if part is not None and (outline := diagram_outline(part.blob)):
                    lines.append(outline)
                    diagrams += 1
            if not p.text.strip():
                continue
            paragraphs.append(p.text)
            lines.append(p.text)
            if p.style and p.style.name and p.style.name.startswith("Heading"):
                try:
                    level = int(p.style.name.split()[-1])
                except ValueError:
                    level = 0
                headings.append({"level": level, "text": p.text.strip()})
        elif tag == "tbl":
            rows = []
            for row in Table(child, doc).rows:
                cells: list[str] = []
                for cell in row.cells:  # merged cells repeat: keep one copy per run
                    t = cell_text(cell.text)
                    if not cells or t != cells[-1]:
                        cells.append(t)
                rows.append(cells)
                lines.append(" | ".join(cells))
            tables.append({"rows": rows})
    text = "\n".join(lines)
    structured = {
        "type": "docx",
        "paragraph_count": len(paragraphs),
        "headings": headings,
        "table_count": len(tables),
        "tables": tables[:50],
        "chart_count": charts,
        "diagram_count": diagrams,
        "figures": figs.summary(found),
    }
    return make_result(text, structured, parser="docx_v1", warnings=warnings)


if __name__ == "__main__":
    cli_run(parse)
