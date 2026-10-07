"""DOCX (modern Word) parser."""

from __future__ import annotations

from pathlib import Path

from ._base import cell_text, cli_run, make_result


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
    lines: list[str] = []  # document order: a table stays under the heading/caption it was written below
    for child in doc.element.body.iterchildren():
        tag = child.tag.rsplit("}", 1)[-1]
        if tag == "p":
            p = Paragraph(child, doc)
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
    }
    return make_result(text, structured, parser="docx_v1")


if __name__ == "__main__":
    cli_run(parse)
