"""Document parsers — one module per format.

Every parser script can be invoked standalone:

    python -m finetune_studio.data.parsers.pdf path/to/file.pdf
    python -m finetune_studio.data.parsers.docx contract.docx
    python -m finetune_studio.data.parsers.xlsx data.xlsx
    ...

Each outputs JSON to stdout:
    {
      "text": "...",              # plain text (for chunking into Q&A)
      "structured": {...},        # format-specific structured content
      "metadata": {
        "parser": "pdf",
        "version": "1",
        "parsed_at": "2026-09-04T10:30:00Z",
        "char_count": 12345,
        "warnings": []
      }
    }

Or as Python:
    from finetune_studio.data.parsers.pdf import parse
    result = parse(Path("file.pdf"))
"""

from __future__ import annotations

import importlib
from pathlib import Path
from typing import Any

# Version of the parser output schema. Bump when format changes.
PARSER_SCHEMA_VERSION = "1"

# Map extension -> (module_name, parse_function_name)
PARSERS: dict[str, tuple[str, str]] = {
    # text / code
    ".txt": ("text", "parse"),
    ".md": ("text", "parse"),
    ".markdown": ("text", "parse"),
    ".log": ("text", "parse"),
    ".py": ("text", "parse"),
    ".js": ("text", "parse"),
    ".ts": ("text", "parse"),
    ".jsx": ("text", "parse"),
    ".tsx": ("text", "parse"),
    ".css": ("text", "parse"),
    ".yaml": ("text", "parse"),
    ".yml": ("text", "parse"),
    ".ini": ("text", "parse"),
    ".cfg": ("text", "parse"),
    ".conf": ("text", "parse"),
    # structured text
    ".csv": ("csv", "parse"),
    ".tsv": ("csv", "parse"),
    ".json": ("json", "parse"),
    ".jsonl": ("jsonl", "parse"),
    ".xml": ("xml", "parse"),
    # markup
    ".html": ("html", "parse"),
    ".htm": ("html", "parse"),
    # images (OCR via tesseract)
    ".png": ("image", "parse"),
    ".jpg": ("image", "parse"),
    ".jpeg": ("image", "parse"),
    ".tif": ("image", "parse"),
    ".tiff": ("image", "parse"),
    ".bmp": ("image", "parse"),
    ".webp": ("image", "parse"),
    ".gif": ("image", "parse"),
    # office
    ".pdf": ("pdf", "parse"),
    ".docx": ("docx", "parse"),
    ".doc": ("doc", "parse"),
    ".xlsx": ("xlsx", "parse"),
    ".xls": ("xls", "parse"),
    ".pptx": ("pptx", "parse"),
    ".odt": ("odf", "parse"),
    ".ods": ("odf", "parse"),
    ".odp": ("odf", "parse"),
    # other
    ".rtf": ("rtf", "parse"),
    ".epub": ("epub", "parse"),
    ".eml": ("email", "parse"),
    ".msg": ("email", "parse"),
}


def readable_extensions() -> frozenset[str]:
    """Extensions the app reads: a parser's, or a format LibreOffice renders (``render_fallback``)."""
    from finetune_studio.data.render import RENDERABLE
    return frozenset(PARSERS) | RENDERABLE


def get_parser_for(path: str | Path) -> Any | None:
    """Returns the parse function for the given path, or None if unsupported."""
    ext = Path(path).suffix.lower()
    info = PARSERS.get(ext)
    if not info:
        return None
    mod_name, fn_name = info
    try:
        mod = importlib.import_module(f".{mod_name}", __name__)
        return getattr(mod, fn_name)
    except (ImportError, AttributeError) as e:
        raise RuntimeError(f"Failed to load parser {mod_name}.{fn_name}: {e}") from e


def parse(path: str | Path) -> dict:
    """Parse a file using the appropriate parser.

    Returns: {"text": str, "structured": dict, "metadata": dict}
    """
    p = Path(path)
    if not p.exists():
        return {
            "text": "",
            "structured": {},
            "metadata": {
                "parser": "none",
                "version": PARSER_SCHEMA_VERSION,
                "char_count": 0,
                "warnings": [f"file not found: {path}"],
            },
        }
    fn = get_parser_for(p)
    if fn is None and _is_binary(p):
        # no parser and not text: open it with LibreOffice and read its pages (was: read as "text" = binary garbage)
        return render_fallback(p, f"no parser for {p.suffix or 'files without an extension'}")
    if fn is None:
        # Fallback: try reading as text
        try:
            text = p.read_text(encoding="utf-8", errors="replace")
            return {
                "text": text,
                "structured": {"raw": text, "fallback": "text"},
                "metadata": {
                    "parser": "text_fallback",
                    "version": PARSER_SCHEMA_VERSION,
                    "char_count": len(text),
                    "warnings": [f"unknown extension {p.suffix}; read as plain text"],
                },
            }
        except OSError as e:
            return {
                "text": "",
                "structured": {},
                "metadata": {
                    "parser": "text_fallback",
                    "version": PARSER_SCHEMA_VERSION,
                    "char_count": 0,
                    "warnings": [f"unknown extension {p.suffix} and text read failed: {e}"],
                },
            }
    try:
        result = fn(p)
    except Exception as e:  # any parser crash: the rendered pages are the next best reading
        if p.suffix.lower() in NO_RENDER_FALLBACK:
            raise
        return render_fallback(p, f"the {p.suffix} parser failed ({type(e).__name__}: {e})")
    if p.suffix.lower() not in NO_RENDER_FALLBACK and _thin(result.get("text") or ""):
        fallback = render_fallback(p, f"the {p.suffix} parser found almost no text", parsed=result)
        if fallback.get("text"):
            return fallback
        result.setdefault("metadata", {}).setdefault("warnings", []).extend(fallback["metadata"]["warnings"])
    return result


# images and PDFs read their own pages (figures); text formats have nothing to render
NO_RENDER_FALLBACK = frozenset({".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp", ".webp", ".gif", ".pdf", ".txt", ".md",
                                ".markdown", ".log", ".py", ".js", ".ts", ".jsx", ".tsx", ".css", ".yaml", ".yml", ".ini",
                                ".cfg", ".conf", ".csv", ".tsv", ".json", ".jsonl", ".xml", ".html", ".htm", ".eml", ".msg"})
THIN_CHARS = 40


def _thin(text: str) -> bool:
    return len(text.strip()) < THIN_CHARS


def _is_binary(p: Path) -> bool:
    try:
        head = p.read_bytes()[:8192]
    except OSError:
        return False
    if b"\x00" in head:
        return True
    try:
        head.decode("utf-8")
    except UnicodeDecodeError as e:
        return e.start < len(head) - 4    # a multi-byte character cut by the 8 KiB window is still text
    return False


def render_fallback(p: Path, reason: str, parsed: dict | None = None) -> dict:
    """Read ``p`` through LibreOffice: every page a figure block (``figures.rendered``). On failure an empty result whose
    warnings say why (never binary bytes as text)."""
    from finetune_studio.data import figures
    from finetune_studio.data.ocr import DEFAULT_LANGS
    from finetune_studio.data.render import RenderError

    from ._base import make_result

    warnings = [f"{reason}: read through LibreOffice, page by page"]
    try:
        got = figures.rendered(p)
    except RenderError as e:
        return make_result("", {"type": "rendered", "error": str(e)}, parser="render_v1_failed",
                           warnings=[f"{reason}; LibreOffice could not open it either: {e}"])
    blocks = [figures.rendered_block(f, t, DEFAULT_LANGS, warnings) for f, t in zip(got.figures, got.texts, strict=True)]
    if got.pages > len(got.figures):
        warnings.append(f"only the first {len(got.figures)} of {got.pages} pages were read (FTS_RENDER_MAX_PAGES)")
    structured = {"type": "rendered", "rendered_from": p.suffix.lower(), "page_count": got.pages,
                  "figures": figures.summary(got.figures)}
    if parsed:
        structured["parser_result"] = parsed.get("metadata", {}).get("parser", "")
    return make_result("\n\n".join(blocks), structured, parser="render_v1_libreoffice", warnings=warnings)


def parse_bytes(filename: str, data: bytes) -> dict:
    """Parse from in-memory bytes (e.g. an upload). Writes to temp file, dispatches."""
    import tempfile
    ext = Path(filename).suffix.lower()
    with tempfile.NamedTemporaryFile(suffix=ext, delete=False) as tmp:
        tmp.write(data)
        tmp_path = Path(tmp.name)
    try:
        result = parse(tmp_path)
        # Carry the original filename in metadata so callers can attribute the result
        result.setdefault("metadata", {})["source_filename"] = filename
        return result
    finally:
        try:
            tmp_path.unlink()
        except OSError:
            pass
