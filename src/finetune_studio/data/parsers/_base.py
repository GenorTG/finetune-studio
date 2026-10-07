"""Shared helpers for parser scripts."""

from __future__ import annotations

import datetime as _dt
import json as _json
import sys
from pathlib import Path

SCHEMA_VERSION = "1"


def now_iso() -> str:
    return _dt.datetime.now(_dt.UTC).isoformat(timespec="seconds")


def cell_text(value: object) -> str:
    """One spreadsheet/table cell as the text a person reads.

    Whole-number floats lose their ``.0`` (an extension 2100 is not "2100.0", 412 plugs are not "412.0"), dates are ISO, and a
    newline inside a cell becomes a space — otherwise the row text breaks in two and the second half lands in another chunk.
    """
    if value is None:
        return ""
    if isinstance(value, bool):
        return "TRUE" if value else "FALSE"
    if isinstance(value, float):
        return str(int(value)) if value.is_integer() and abs(value) < 1e15 else repr(value)
    if isinstance(value, _dt.datetime):
        return value.date().isoformat() if value.time() == _dt.time(0) else value.isoformat(sep=" ")
    if isinstance(value, _dt.date):
        return value.isoformat()
    return " ".join(str(value).split())


def make_result(text: str, structured: dict, parser: str, warnings: list[str] | None = None, **extra) -> dict:
    """Build the standard parser output envelope."""
    meta = {
        "parser": parser,
        "version": SCHEMA_VERSION,
        "parsed_at": now_iso(),
        "char_count": len(text or ""),
        "warnings": list(warnings or []),
    }
    meta.update(extra)
    return {
        "text": text or "",
        "structured": structured or {},
        "metadata": meta,
    }


def cli_run(parse_fn) -> None:
    """Common __main__ entry: read path from argv, call parse_fn, print JSON to stdout.

    Usage:
        if __name__ == "__main__":
            from ._base import cli_run
            cli_run(parse)
    """
    if len(sys.argv) < 2:
        print(f"usage: {sys.argv[0]} <path-to-file> [--pretty]", file=sys.stderr)
        sys.exit(2)
    path = Path(sys.argv[1])
    pretty = "--pretty" in sys.argv
    try:
        result = parse_fn(path)
    except Exception as e:  # noqa: BLE001 - CLI boundary: any parser failure becomes a JSON error + exit 1
        print(_json.dumps({"error": str(e), "parser": parse_fn.__name__}), file=sys.stderr)
        sys.exit(1)
    indent = 2 if pretty else None
    _json.dump(result, sys.stdout, indent=indent, ensure_ascii=False)
    sys.stdout.write("\n")
