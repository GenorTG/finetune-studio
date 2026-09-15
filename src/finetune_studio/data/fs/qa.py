"""Q&A pair + source storage on disk (parallel to curated.db for portability).

LAYOUT (per project):
  qa/pairs/<qa-id>.json     — individual Q&A record
  qa/sources/<source-id>.json — what produced the pairs (file, slice, etc.)
"""
from __future__ import annotations

import hashlib
import json
import mimetypes
import time
import uuid
from pathlib import Path

from finetune_studio.data.fs.paths import project_dir


def register_qa_source(
    pid: str,
    path: str,
    *,
    mime_type: str = "",
    filename: str | None = None,
) -> dict:
    """Register a file path as a QA source (idempotent on ``pid`` + path).

    Used by ``POST /api/projects/{pid}/data-prep/sources`` to promote a
    file-library upload into the data-prep source picker without re-uploading.

    Always runs parse+chunk (shared with DataPrepRunner) so ``read_source``
    finds ``files/<sha12>/parsed.txt``. Existing ready sources are returned
    as-is; existing-but-unparsed sources are re-ingested.
    """
    from finetune_studio.data.prep.ingest import ensure_qa_source_parsed

    p = Path(path)
    abs_path = str(p.resolve()) if p.exists() else str(p)
    for existing in list_qa_sources(pid):
        if existing.get("data_path") == abs_path or existing.get("path") == abs_path:
            if (
                existing.get("status") == "ready"
                and int(existing.get("chunk_count") or 0) > 0
            ):
                return existing
            return ensure_qa_source_parsed(pid, existing)

    name = filename or (p.name if p.name else "upload")
    mime = mime_type or (mimetypes.guess_type(name)[0] or "")
    sha256 = ""
    char_count = 0
    size_bytes = 0
    if p.is_file():
        raw = p.read_bytes()
        sha256 = hashlib.sha256(raw).hexdigest()
        size_bytes = len(raw)
        char_count = len(raw.decode("utf-8", errors="ignore"))
    source_id = sha256[:12] if sha256 else uuid.uuid4().hex[:12]
    source: dict = {
        "id": source_id,
        "sha256": sha256,
        "filename": name,
        "name": name,
        "mime_type": mime,
        "char_count": char_count,
        "chunk_count": 0,
        "parser": "",
        "uploaded_at": time.time(),
        "status": "registered",
        "data_path": abs_path,
        "path": abs_path,
        "size_bytes": size_bytes,
    }
    write_qa_source(pid, source)
    return ensure_qa_source_parsed(pid, source)


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
    except Exception:  # noqa: BLE001
        return {}


def list_qa_pairs(pid: str, source_id: str | None = None, status: str | None = None) -> list[dict]:
    pairs_dir = project_dir(pid) / "qa" / "pairs"
    if not pairs_dir.exists():
        return []
    out = []
    for p in sorted(pairs_dir.glob("*.json")):
        try:
            qa = json.loads(p.read_text(encoding="utf-8"))
        except Exception:  # noqa: BLE001, S112
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
        except Exception:  # noqa: BLE001, S112
            continue
    return sorted(out, key=lambda x: x.get("uploaded_at", 0), reverse=True)


def update_qa_pair(pid: str, qa_id: str, **fields) -> dict | None:
    p = project_dir(pid) / "qa" / "pairs" / f"{qa_id}.json"
    if not p.exists():
        return None
    try:
        qa = json.loads(p.read_text(encoding="utf-8"))
    except Exception:  # noqa: BLE001
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
            except Exception:  # noqa: BLE001, S112
                continue
    if src.exists():
        src.unlink()
        return True
    return False
