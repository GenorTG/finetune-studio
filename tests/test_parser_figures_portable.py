"""Figure extraction on hosts without tesseract or poppler (CI): parses still succeed or fail with a reason."""
from __future__ import annotations

import io
from pathlib import Path

import pytest

from finetune_studio.data import figures


def _png() -> bytes:
    from PIL import Image
    out = io.BytesIO()
    Image.new("RGB", (400, 300), "white").save(out, "PNG")
    return out.getvalue()


def _text_pdf_page(text: str) -> bytes:
    from tests.test_parser_figures import _text_pdf_page as make
    return make(text)


def test_without_poppler_a_scanned_pdf_parses_with_a_warning(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    # CI has no poppler: the figure extractor's RuntimeError used to escape the PDF parser
    from PIL import Image
    from pypdf import PdfReader, PdfWriter

    from finetune_studio.data.parsers import pdf as parser

    scan = io.BytesIO()
    Image.open(io.BytesIO(_png())).convert("RGB").save(scan, "PDF", resolution=150)
    w = PdfWriter()
    for src in (_text_pdf_page("A digital page with enough real text on it to count."), scan.getvalue()):
        for page in PdfReader(io.BytesIO(src)).pages:
            w.add_page(page)
    p = tmp_path / "mixed.pdf"
    with p.open("wb") as f:
        w.write(f)
    monkeypatch.setattr(figures.shutil, "which", lambda name: None)
    out = parser.parse(p)
    assert "enough real text" in out["text"]
    assert any("not read" in x and "pdftoppm" in x for x in out["metadata"]["warnings"])


def test_a_broken_image_is_no_figure() -> None:
    import tempfile
    with tempfile.NamedTemporaryFile(suffix=".png") as f:
        f.write(b"\x89PNG\r\n\x1a\nnot-really")
        f.flush()
        from finetune_studio.data.parsers import image
        out = image.parse(Path(f.name))
    assert out["text"] == "" and "not a readable image" in out["metadata"]["warnings"]
