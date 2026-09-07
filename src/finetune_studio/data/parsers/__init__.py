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
from typing import Any, Optional

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


def list_parsers() -> list[dict]:
    """All supported parsers + which one handles which extension."""
    out = []
    for ext, (mod, fn) in sorted(PARSERS.items()):
        out.append({"extension": ext, "module": mod, "function": fn})
    return out


def get_parser_for(path: str | Path) -> Optional[Any]:
    """Returns the parse function for the given path, or None if unsupported."""
    ext = Path(path).suffix.lower()
    info = PARSERS.get(ext)
    if not info:
        return None
    mod_name, fn_name = info
    try:
        mod = importlib.import_module(f".{mod_name}", __name__)
        return getattr(mod, fn_name)
    except Exception as e:
        raise RuntimeError(f"Failed to load parser {mod_name}.{fn_name}: {e}")


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
        except Exception as e:
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
    return fn(p)


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
