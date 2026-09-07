"""project.json — single JSON document with project metadata."""
from __future__ import annotations

import json
from pathlib import Path

from finetune_studio.data.fs.paths import project_dir


def write_project_json(pid: str, data: dict) -> Path:
    p = project_dir(pid) / "project.json"
    p.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")
    return p


def read_project_json(pid: str) -> dict:
    p = project_dir(pid) / "project.json"
    if not p.exists():
        return {}
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except Exception:
        return {}
