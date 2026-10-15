"""The one parse of each library file version: a ``file_conversions`` row with ``format = 'parsed'``.

``knowledge.sources`` owns the meaning; this module is only the SQL. A row exists from the moment a readable file is
staged (``queued``) and moves through ``parsing`` to ``ready`` or ``error``. The parsed text and its passages live on
disk under ``files/<sha12>/`` (``converted_path`` points at ``parsed.txt``).
"""
from __future__ import annotations

import time
from typing import Any

from finetune_studio.db.connection import cursor

FORMAT = "parsed"
_FIELDS = {"converted_path", "converted_hash", "converted_size", "converter", "status", "error_message",
           "char_count", "chunk_count", "edited_at"}

# One row per live file: its current version, the parse row of that version (if any) and the raw hash that keys
# mining items and pairs (``source_id`` = first 12 hex of it).
_SELECT = """
SELECT pf.id AS file_id, pf.project_id, pf.original_name, pf.mime_type, pf.size_bytes, pf.uploaded_at,
       pf.deleted_at, fv.version, fv.raw_path, fv.raw_hash,
       fc.converted_path, fc.converted_hash, fc.converter, fc.converted_at, fc.status, fc.error_message,
       fc.char_count, fc.chunk_count, fc.edited_at
  FROM project_files pf
  JOIN file_versions fv ON fv.file_id = pf.id AND fv.version = pf.current_version
  LEFT JOIN file_conversions fc ON fc.file_id = pf.id AND fc.version = pf.current_version AND fc.format = 'parsed'
"""


def rows(project_id: str, *, include_deleted: bool = False) -> list[dict[str, Any]]:
    """Every file of the project with its current version and parse state, oldest upload first."""
    where = " WHERE pf.project_id = ?" + ("" if include_deleted else " AND pf.deleted_at IS NULL")
    with cursor() as c:
        return [dict(r) for r in c.execute(_SELECT + where + " ORDER BY pf.uploaded_at, pf.id", (project_id,))]


def row_for_file(project_id: str, file_id: str) -> dict[str, Any] | None:
    with cursor() as c:
        r = c.execute(_SELECT + " WHERE pf.project_id = ? AND pf.id = ?", (project_id, file_id)).fetchone()
    return dict(r) if r else None


def row_for_hash(project_id: str, sha12: str, *, include_deleted: bool = False) -> dict[str, Any] | None:
    """The file whose current version's raw bytes hash to ``sha12…`` (a project never holds the same bytes twice)."""
    where = " WHERE pf.project_id = ? AND substr(fv.raw_hash, 1, 12) = ?"
    if not include_deleted:
        where += " AND pf.deleted_at IS NULL"
    with cursor() as c:
        r = c.execute(_SELECT + where + " ORDER BY pf.deleted_at IS NOT NULL LIMIT 1", (project_id, sha12)).fetchone()
    return dict(r) if r else None


def upsert(file_id: str, version: int, **fields: Any) -> None:
    """Create or update the parse row of ``(file_id, version)``; unknown fields are a programming error."""
    bad = set(fields) - _FIELDS
    if bad:
        raise ValueError(f"unknown parse fields: {sorted(bad)}")
    now = time.time()
    base = {"converted_path": "", "converted_hash": "", "converted_size": 0, "converter": "", "status": "queued",
            "error_message": None, "char_count": 0, "chunk_count": 0, "edited_at": None}
    with cursor() as c:
        exists = c.execute("SELECT 1 FROM file_conversions WHERE file_id = ? AND version = ? AND format = ?",
                           (file_id, version, FORMAT)).fetchone()
        if exists:
            if fields:
                sets = ", ".join(f"{k} = ?" for k in fields)
                c.execute(f"UPDATE file_conversions SET {sets}, converted_at = ? "
                          "WHERE file_id = ? AND version = ? AND format = ?",
                          [*fields.values(), now, file_id, version, FORMAT])
            return
        data = {**base, **fields}
        cols = ", ".join(data)
        c.execute(f"INSERT INTO file_conversions (file_id, version, format, converted_at, {cols}) "
                  f"VALUES (?, ?, ?, ?, {', '.join('?' * len(data))})",
                  [file_id, version, FORMAT, now, *data.values()])


def delete(file_id: str) -> None:
    with cursor() as c:
        c.execute("DELETE FROM file_conversions WHERE file_id = ? AND format = ?", (file_id, FORMAT))
