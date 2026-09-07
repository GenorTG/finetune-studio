"""PDF parser. Tries pypdf → PyPDF2 → pdftotext (CLI) → OCR via tesseract."""

from __future__ import annotations

import logging
import subprocess
from pathlib import Path

from ..ocr import DEFAULT_LANGS, ocr_pdf
from ._base import cli_run, make_result

log = logging.getLogger(__name__)


def parse(path: Path, languages: str = DEFAULT_LANGS) -> dict:
    """Parse a PDF into plain text. Try cheap text extractors first, OCR as fallback.

    Returns the standard parser envelope. metadata.parser indicates the final
    method (e.g. "pdf_v1_ocr_eng+pol" if OCR was the path taken).
    """
    warnings: list[str] = []
    pages_text, n_pages, method = _extract_with_pypdf(path)
    if not pages_text:
        pages_text, n_pages, method = _extract_with_pypdf2(path)
    if not pages_text:
        pages_text, n_pages, method = _extract_with_pdftotext(path)
    if not pages_text:
        # Last resort: OCR each page as an image.
        try:
            ocr_pages = ocr_pdf(path, languages=languages)
            pages_text = [p["text"] for p in ocr_pages if p.get("text")]
            n_pages = len(ocr_pages)
            method = f"ocr_{languages}"
            warnings.append(f"PDF had no extractable text; ran OCR ({languages})")
        except Exception as e:  # noqa: BLE001
            warnings.append(f"OCR fallback also failed: {e}")
            method = "ocr-failed"
            n_pages = 0
    text = "\n\n".join(pages_text) if pages_text else ""
    structured = {
        "type": "pdf",
        "page_count": n_pages,
        "extraction_method": method,
        "pages": [{"page": i + 1, "text": p} for i, p in enumerate(pages_text[:500])],
        "languages": languages,
    }
    if not pages_text:
        warnings.append("No text extracted — install pypdf, poppler (pdftotext), or tesseract")
    return make_result(text, structured, parser=f"pdf_v1_{method}", warnings=warnings)


def _extract_with_pypdf(path: Path):
    try:
        from pypdf import PdfReader
    except ImportError:
        return [], 0, "pypdf-missing"
    try:
        reader = PdfReader(str(path))
        pages = [(p.extract_text() or "").strip() for p in reader.pages]
        return [p for p in pages if p], len(reader.pages), "pypdf"
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
        return [p for p in pages if p], len(reader.pages), "pypdf2"
    except Exception as e:  # noqa: BLE001
        return [], 0, f"pypdf2-error:{e}"


def _extract_with_pdftotext(path: Path):
    try:
        r = subprocess.run(["pdftotext", str(path), "-"], capture_output=True, text=True, timeout=120, check=False)
        if r.returncode == 0 and r.stdout.strip():
            pages = r.stdout.split("\x0c")  # form-feed between pages
            pages = [(p or "").strip() for p in pages]
            return [p for p in pages if p], len(pages), "pdftotext"
    except (FileNotFoundError, subprocess.TimeoutExpired):
        pass
    return [], 0, "pdftotext-empty"


if __name__ == "__main__":
    cli_run(parse)
