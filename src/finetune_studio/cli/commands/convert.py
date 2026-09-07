"""`fts convert` — convert between data formats (csv ↔ jsonl ↔ json)."""
from __future__ import annotations

import sys
from pathlib import Path


def cmd_convert(args) -> None:
    from finetune_studio.data.converter import (
        csv_to_jsonl,
        json_to_jsonl,
        jsonl_to_json,
    )

    src = Path(args.source)
    if not src.exists():
        print(f"Error: File not found: {src}")
        sys.exit(1)

    target = args.output or str(src.with_suffix(f".{args.target_format}"))

    if src.suffix == ".jsonl" and args.target_format == "json":
        jsonl_to_json(str(src), target)
    elif src.suffix == ".json" and args.target_format == "jsonl":
        json_to_jsonl(str(src), target)
    elif src.suffix == ".csv" and args.target_format == "jsonl":
        csv_to_jsonl(str(src), target, system_prompt=args.system_prompt)
    else:
        print(f"Error: Cannot convert {src.suffix} -> .{args.target_format}")
        sys.exit(1)

    print(f"Converted: {src} -> {target}")
