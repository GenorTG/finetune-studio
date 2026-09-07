"""Shared helpers for parser scripts."""

from __future__ import annotations

import datetime as _dt
import json as _json
import sys
from pathlib import Path
from typing import Any

SCHEMA_VERSION = "1"


def now_iso() -> str:
    return _dt.datetime.now(_dt.timezone.utc).isoformat(timespec="seconds")


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
    except Exception as e:
        print(_json.dumps({"error": str(e), "parser": parse_fn.__name__}), file=sys.stderr)
        sys.exit(1)
    indent = 2 if pretty else None
    _json.dump(result, sys.stdout, indent=indent, ensure_ascii=False)
    sys.stdout.write("\n")
