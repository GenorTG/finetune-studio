"""Q&A pair + source storage on disk (parallel to curated.db for portability).

LAYOUT (per project):
  qa/pairs/<qa-id>.json     — individual Q&A record
  qa/sources/<source-id>.json — what produced the pairs (file, slice, etc.)
"""
from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Optional

from finetune_studio.data.fs.paths import project_dir


def write_qa_pair(pid: str, qa: dict) -> None:
    qa_dir = project_dir(pid) / "qa" / "pairs"
    qa_dir.mkdir(parents=True, exist_ok=True)
    (qa_dir / f"{qa['id']}.json").write_text(json.dumps(qa, indent=2, ensure_ascii=False), encoding="utf-8")


def write_qa_source(pid: str, source: dict) -> None:
    src_dir = project_dir(pid) / "qa" / "sources"
    src_dir.mkdir(parents=True, exist_ok=True)
    (src_dir / f"{source['id']}.json").write_text(json.dumps(source, indent=2, ensure_ascii=False), encoding="utf-8")


def read_qa_source(pid: str, source_id: str) -> dict:
    p = project_dir(pid) / "qa" / "sources" / f"{source_id}.json"
    if not p.exists():
        return {}
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except Exception:
        return {}


def list_qa_pairs(pid: str, source_id: Optional[str] = None, status: Optional[str] = None) -> list[dict]:
    pairs_dir = project_dir(pid) / "qa" / "pairs"
    if not pairs_dir.exists():
        return []
    out = []
    for p in sorted(pairs_dir.glob("*.json")):
        try:
            qa = json.loads(p.read_text(encoding="utf-8"))
        except Exception:
            continue
        if source_id and qa.get("source_id") != source_id:
            continue
        if status and qa.get("status") != status:
            continue
        out.append(qa)
    return out


def list_qa_sources(pid: str) -> list[dict]:
    src_dir = project_dir(pid) / "qa" / "sources"
    if not src_dir.exists():
        return []
    out = []
    for p in src_dir.glob("*.json"):
        try:
            out.append(json.loads(p.read_text(encoding="utf-8")))
        except Exception:
            continue
    return sorted(out, key=lambda x: x.get("uploaded_at", 0), reverse=True)


def update_qa_pair(pid: str, qa_id: str, **fields) -> Optional[dict]:
    p = project_dir(pid) / "qa" / "pairs" / f"{qa_id}.json"
    if not p.exists():
        return None
    try:
        qa = json.loads(p.read_text(encoding="utf-8"))
    except Exception:
        return None
    for k, v in fields.items():
        qa[k] = v
    qa["updated_at"] = time.time()
    p.write_text(json.dumps(qa, indent=2, ensure_ascii=False), encoding="utf-8")
    return qa


def delete_qa_source(pid: str, source_id: str) -> bool:
    """Delete a source AND all its Q&A pairs (filesystem-side)."""
    src = project_dir(pid) / "qa" / "sources" / f"{source_id}.json"
    pairs = project_dir(pid) / "qa" / "pairs"
    if pairs.exists():
        for p in list(pairs.glob("*.json")):
            try:
                qa = json.loads(p.read_text(encoding="utf-8"))
                if qa.get("source_id") == source_id:
                    p.unlink()
            except Exception:
                continue
    if src.exists():
        src.unlink()
        return True
    return False
