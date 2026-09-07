"""Writer for parsed.txt + parsed.json (the post-parse deliverables for one file)."""
from __future__ import annotations

import json

from finetune_studio.data.fs.paths import file_dir


def write_parsed_outputs(pid: str, sha256: str, parsed_text: str, parsed_json: dict) -> None:
    fd = file_dir(pid, sha256)
    (fd / "parsed.txt").write_text(parsed_text, encoding="utf-8")
    (fd / "parsed.json").write_text(json.dumps(parsed_json, indent=2, ensure_ascii=False), encoding="utf-8")
