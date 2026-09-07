"""PPTX (PowerPoint) parser."""

from __future__ import annotations

from pathlib import Path

from ._base import cli_run, make_result


def parse(path: Path) -> dict:
    try:
        from pptx import Presentation
    except ImportError as e:
        return make_result("", {"type": "pptx", "error": str(e)}, parser="pptx_v1",
                           warnings=["install python-pptx (pip install python-pptx)"])
    prs = Presentation(str(path))
    text_lines = []
    slides_structured = []
    for i, slide in enumerate(prs.slides, 1):
        texts = []
        for shape in slide.shapes:
            if hasattr(shape, "text") and shape.text.strip():
                texts.append(shape.text)
            if shape.has_table:
                for row in shape.table.rows:
                    texts.append(" | ".join(cell.text.strip() for cell in row.cells))
        slide_text = "\n".join(texts)
        text_lines.append(f"=== Slide {i} ===\n{slide_text}")
        slides_structured.append({"slide": i, "shape_count": len(slide.shapes), "text": slide_text})
    text = "\n\n".join(text_lines)
    structured = {"type": "pptx", "slide_count": len(prs.slides), "slides": slides_structured}
    return make_result(text, structured, parser="pptx_v1")


if __name__ == "__main__":
    cli_run(parse)
