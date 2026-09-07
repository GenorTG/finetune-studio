"""Content-addressed file storage.

Single responsibility: store uploaded bytes under files/<sha>/<original-filename>,
deal with re-uploads (rename + alias tracking), and list/delete.

If the same content arrives later under a different name:
- the on-disk file is renamed to the new name
- the old name is preserved in metadata.aliases

Idempotent — re-uploading identical bytes is a no-op except for last_seen_at
+ alias tracking.
"""
from __future__ import annotations

import json
import shutil
import time
from pathlib import Path

from finetune_studio.data.fs.metadata import (
    FileMetadata,
    _safe_filename,
    hash_bytes,
    read_file_metadata,
    update_file_metadata,
)
from finetune_studio.data.fs.paths import file_dir, project_dir


def store_file(
    pid: str,
    data: bytes,
    *,
    original_filename: str,
    mime_type: str = "",
    uploaded_by: str = "",
    source_kind: str = "upload",
    notes: str = "",
) -> tuple[Path, FileMetadata]:
    """Store uploaded bytes content-addressed, preserving the ORIGINAL filename on disk."""
    sha = hash_bytes(data)
    fd = file_dir(pid, sha)
    safe_name = _safe_filename(original_filename)
    ext = Path(safe_name).suffix.lower() or ""
    canonical = fd / safe_name
    existing_files = [p for p in fd.iterdir() if p.is_file() and p.suffix.lower() == ext] if fd.exists() else []
    now = time.time()

    if existing_files:
        # Rename existing file to the new name so the disk reflects what the
        # user most recently called it.
        if existing_files[0].name != safe_name:
            try:
                existing_files[0].rename(canonical)
            except OSError:
                pass  # cross-device or other rename failure — fall through to write
    if not canonical.exists() or canonical.stat().st_size != len(data):
        canonical.write_bytes(data)

    # Metadata: merge with existing if present.
    meta_path = fd / "metadata.json"
    if meta_path.exists():
        try:
            existing = json.loads(meta_path.read_text(encoding="utf-8"))
        except Exception:
            existing = {}
        aliases = list(existing.get("aliases", []))
        old_canonical = existing.get("original_filename")
        if old_canonical and old_canonical != safe_name and old_canonical not in aliases:
            aliases.append(old_canonical)
        if safe_name != old_canonical and safe_name not in aliases:
            aliases.append(safe_name)
        existing.update({
            "sha256": sha,
            "original_filename": safe_name,
            "mime_type": mime_type or existing.get("mime_type", ""),
            "ext": ext,
            "byte_count": len(data),
            "uploaded_by": uploaded_by or existing.get("uploaded_by", ""),
            "source_kind": source_kind or existing.get("source_kind", "upload"),
            "notes": notes or existing.get("notes", ""),
            "aliases": aliases,
            "last_seen_at": now,
        })
        meta = FileMetadata(**{k: existing.get(k, v) for k, v in FileMetadata.__dataclass_fields__.items()})
    else:
        meta = FileMetadata(
            sha256=sha,
            original_filename=safe_name,
            mime_type=mime_type,
            ext=ext,
            byte_count=len(data),
            uploaded_at=now,
            last_seen_at=now,
            uploaded_by=uploaded_by,
            source_kind=source_kind,
            notes=notes,
            aliases=[safe_name],
        )
    meta_path.write_text(json.dumps(meta.to_json(), indent=2, ensure_ascii=False), encoding="utf-8")
    return fd, meta


def list_files(pid: str) -> list[FileMetadata]:
    out = []
    files_root = project_dir(pid) / "files"
    if not files_root.exists():
        return out
    for sha_dir in sorted(files_root.iterdir()):
        if not sha_dir.is_dir():
            continue
        m = read_file_metadata(pid, sha_dir.name)
        if m:
            out.append(m)
    return out


def delete_file(pid: str, sha256: str) -> bool:
    fd = file_dir(pid, sha256)
    if fd.exists():
        shutil.rmtree(fd)
        return True
    return False
