"""JSON parser."""

from __future__ import annotations

import json
from pathlib import Path

from ._base import cli_run, make_result


def parse(path: Path) -> dict:
    raw = path.read_text(encoding="utf-8", errors="replace")
    data = json.loads(raw) if raw.strip() else None
    # Pretty-printed JSON is great AI-readable text
    text = json.dumps(data, indent=2, ensure_ascii=False) if data is not None else ""
    structured = {
        "type": "json",
        "schema": _type_of(data),
        "size_bytes": len(raw),
    }
    return make_result(text, structured, parser="json_v1")


def _type_of(obj) -> str:
    if obj is None: return "null"
    if isinstance(obj, bool): return "boolean"
    if isinstance(obj, (int, float)): return "number"
    if isinstance(obj, str): return "string"
    if isinstance(obj, list): return "array"
    if isinstance(obj, dict): return "object"
    return "unknown"


if __name__ == "__main__":
    cli_run(parse)
