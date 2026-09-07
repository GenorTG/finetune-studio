"""FileMetadata dataclass + filename safety + read/update helpers.

Single responsibility: describe and persist metadata for one stored file.
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Optional

from finetune_studio.data.fs.paths import file_dir


@dataclass
class FileMetadata:
    sha256: str
    original_filename: str            # most recent name seen
    mime_type: str
    ext: str
    byte_count: int
    char_count: int = 0
    chunk_count: int = 0
    parser: str = ""  # e.g. "pdf_v1_ocr_pytesseract"
    parser_version: str = ""
    uploaded_at: float = 0.0           # first time this content was seen
    last_seen_at: float = 0.0         # most recent upload
    uploaded_by: str = ""             # user id if known
    source_kind: str = "upload"       # upload | api | batch
    notes: str = ""
    warnings: list = field(default_factory=list)
    # ALL filenames this content was ever uploaded under. Preserves info
    # the user cares about (Acme Termination Policy 2024.pdf, etc.) even
    # when the same bytes arrive later under a different name.
    aliases: list = field(default_factory=list)

    def to_json(self) -> dict:
        return asdict(self)


def hash_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _safe_filename(name: str) -> str:
    """Strip path separators and other dangerous chars from an upload filename."""
    name = name.replace("/", "_").replace("\\", "_").replace("\x00", "_").strip()
    return name or "upload"


def read_file_metadata(pid: str, sha256: str) -> Optional[FileMetadata]:
    p = file_dir(pid, sha256) / "metadata.json"
    if not p.exists():
        return None
    try:
        d = json.loads(p.read_text(encoding="utf-8"))
        return FileMetadata(**d)
    except Exception:
        return None


def update_file_metadata(pid: str, sha256: str, **fields) -> Optional[FileMetadata]:
    """Update specific fields (e.g. char_count, chunk_count, parser) and persist."""
    meta = read_file_metadata(pid, sha256)
    if meta is None:
        return None
    for k, v in fields.items():
        if hasattr(meta, k):
            setattr(meta, k, v)
    p = file_dir(pid, sha256) / "metadata.json"
    p.write_text(json.dumps(meta.to_json(), indent=2, ensure_ascii=False), encoding="utf-8")
    return meta
