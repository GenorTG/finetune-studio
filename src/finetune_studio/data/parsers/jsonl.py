"""JSON Lines parser (one JSON object per line)."""

from __future__ import annotations

import json
from pathlib import Path

from ._base import cli_run, make_result


def parse(path: Path) -> dict:
    parts = []
    records = []
    warnings = []
    with path.open("r", encoding="utf-8", errors="replace") as f:
        for i, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
                records.append(obj)
                parts.append(json.dumps(obj, indent=2, ensure_ascii=False))
            except json.JSONDecodeError as e:
                warnings.append(f"line {i}: {e}")
                parts.append(line)  # keep the raw line
    text = "\n\n".join(parts)
    structured = {
        "type": "jsonl",
        "record_count": len(records),
        "first_record_keys": list(records[0].keys()) if records and isinstance(records[0], dict) else None,
    }
    return make_result(text, structured, parser="jsonl_v1", warnings=warnings)


if __name__ == "__main__":
    cli_run(parse)
