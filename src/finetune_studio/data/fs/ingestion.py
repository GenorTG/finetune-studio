"""Append-only audit log at logs/ingestions.jsonl per project.

Single responsibility: durable, append-only event log of every project action.
"""
from __future__ import annotations

import json
import time

from finetune_studio.data.fs.paths import project_dir


def log_ingestion(pid: str, event: dict) -> None:
    """Append an event to logs/ingestions.jsonl. Creates dir if needed."""
    logs_dir = project_dir(pid) / "logs"
    logs_dir.mkdir(parents=True, exist_ok=True)
    log_path = logs_dir / "ingestions.jsonl"
    if "timestamp" not in event:
        event["timestamp"] = time.time()
    with log_path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(event, ensure_ascii=False) + "\n")


def read_ingestion_log(pid: str, limit: int = 200) -> list[dict]:
    log_path = project_dir(pid) / "logs" / "ingestions.jsonl"
    if not log_path.exists():
        return []
    out = []
    lines = log_path.read_text(encoding="utf-8").splitlines()
    for line in lines[-limit:]:
        line = line.strip()
        if not line:
            continue
        try:
            out.append(json.loads(line))
        except Exception:
            continue
    return out
