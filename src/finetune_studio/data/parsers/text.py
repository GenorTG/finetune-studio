"""Plain text / code / config parser.

Handles: .txt .md .log .py .js .ts .jsx .tsx .css .yaml .yml .ini .cfg .conf
"""

from __future__ import annotations

from pathlib import Path

from ._base import cli_run, make_result


def parse(path: Path) -> dict:
    text = path.read_text(encoding="utf-8", errors="replace")
    structured = {
        "type": "text",
        "line_count": text.count("\n") + (0 if text.endswith("\n") or not text else 1),
        "language": _guess_language(path),
    }
    return make_result(text, structured, parser="text_v1")


def _guess_language(path: Path) -> str:
    ext = path.suffix.lower()
    return {
        ".py": "python", ".js": "javascript", ".ts": "typescript",
        ".jsx": "javascript-jsx", ".tsx": "typescript-jsx",
        ".css": "css", ".yaml": "yaml", ".yml": "yaml",
        ".ini": "ini", ".cfg": "ini", ".conf": "ini",
        ".md": "markdown", ".markdown": "markdown", ".log": "log",
    }.get(ext, "text")


if __name__ == "__main__":
    cli_run(parse)
