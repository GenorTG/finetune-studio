"""PDF parser. Tries pypdf → PyPDF2 → pdftotext (CLI) → OCR via tesseract."""

from __future__ import annotations

import logging
import subprocess
from pathlib import Path

from .. import figures
from ..ocr import DEFAULT_LANGS, ocr_pdf
from ._base import cli_run, make_result

log = logging.getLogger(__name__)


def parse(path: Path, languages: str = DEFAULT_LANGS) -> dict:
    """Parse a PDF into plain text. Try cheap text extractors first, OCR as fallback.

    Returns the standard parser envelope. metadata.parser indicates the final
    method (e.g. "pdf_v1_ocr_eng+pol" if OCR was the path taken).
    """
    warnings: list[str] = []
    first = pages_text, n_pages, method = _extract_with_pypdf(path)
    if not any(pages_text):
        pages_text, n_pages, method = _extract_with_pypdf2(path)
    if not any(pages_text):
        pages_text, n_pages, method = _extract_with_pdftotext(path)
    if not any(pages_text) and first[1]:
        pages_text, n_pages, method = first  # a fully scanned PDF: every page becomes a page figure below
    if not n_pages:
        # no extractor could even count the pages: OCR every page as before
        pages_text, n_pages, method, found = _ocr_all(path, languages, warnings)
    else:
        # A scanned page inside a digital PDF, or a table/chart pasted as a picture, has no text: each becomes a figure
        # block (OCR now, the vision reader during mining). Before, OCR ran only when the WHOLE file had no text.
        try:
            found = figures.extract(path, pages_text) if method in ("pypdf", "pypdf2", "pdftotext") else []
        except RuntimeError as e:  # pdftoppm (poppler) missing: the text layer still stands, say what was not read
            warnings.append(f"pictures and scanned pages not read: {e}")
            found = []
        pages_text = list(pages_text) + [""] * max(0, n_pages - len(pages_text))
        for f in found:
            b = figures.ocr_block(f, languages, warnings)
            i = f.anchor - 1
            if 0 <= i < len(pages_text):
                pages_text[i] = b if f.whole_page else f"{pages_text[i]}\n{b}"
        if any(f.whole_page for f in found):
            method = f"{method}+ocr_{languages}"
    # Blank pages stay in pages_text so structured page numbers match the PDF.
    text = "\n\n".join(p for p in pages_text if p)
    structured = {
        "type": "pdf",
        "page_count": n_pages,
        "extraction_method": method,
        "pages": [{"page": i + 1, "text": p} for i, p in enumerate(pages_text[:500]) if p],
        "languages": languages,
        "figures": figures.summary(found),
    }
    if not any(pages_text):
        warnings.append("No text extracted — install pypdf, poppler (pdftotext), or tesseract")
    return make_result(text, structured, parser=f"pdf_v1_{method}", warnings=warnings)


def _ocr_all(path: Path, languages: str, warnings: list[str]):
    """No text layer anywhere: OCR every page (``figures`` has nothing to anchor to without a readable page list)."""
    try:
        ocr_pages = ocr_pdf(path, languages=languages)
    except Exception as e:  # noqa: BLE001
        warnings.append(f"OCR fallback also failed: {e}")
        return [], 0, "ocr-failed", []
    warnings.append(f"PDF had no extractable text; ran OCR ({languages})")
    return [p.get("text") or "" for p in ocr_pages], len(ocr_pages), f"ocr_{languages}", []


def _extract_with_pypdf(path: Path):
    try:
        from pypdf import PdfReader
    except ImportError:
        return [], 0, "pypdf-missing"
    try:
        reader = PdfReader(str(path))
        pages = [(p.extract_text() or "").strip() for p in reader.pages]
        return pages, len(reader.pages), "pypdf"
    except Exception as e:  # noqa: BLE001
        return [], 0, f"pypdf-error:{e}"


def _extract_with_pypdf2(path: Path):
    try:
        from PyPDF2 import PdfReader
    except ImportError:
        return [], 0, "pypdf2-missing"
    try:
        reader = PdfReader(str(path))
        pages = [(p.extract_text() or "").strip() for p in reader.pages]
        return pages, len(reader.pages), "pypdf2"
    except Exception as e:  # noqa: BLE001
        return [], 0, f"pypdf2-error:{e}"


def _extract_with_pdftotext(path: Path):
    try:
        r = subprocess.run(["pdftotext", str(path), "-"], capture_output=True, text=True, errors="replace", timeout=120, check=False)
        if r.returncode == 0 and r.stdout.strip():
            pages = r.stdout.split("\x0c")  # form-feed between pages
            if pages and not pages[-1].strip():
                pages.pop()  # pdftotext ends with a form feed
            pages = [p.strip() for p in pages]
            return pages, len(pages), "pdftotext"
    except (OSError, subprocess.TimeoutExpired):
        pass
    return [], 0, "pdftotext-empty"


if __name__ == "__main__":
    cli_run(parse)
