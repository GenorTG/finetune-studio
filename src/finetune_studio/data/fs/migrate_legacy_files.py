"""Migrate legacy filesystem files into the new project_files DB table.

Reads metadata.json from each file directory under
~/.finetune-studio/projects/{pid}/files/{hash}/ and inserts a row
into project_files so the file library UI can show them.

Usage:
    python -m finetune_studio.data.fs.migrate_legacy_files [project_id]

If no project_id given, migrates ALL projects that have a files/ dir.
"""
from __future__ import annotations

import json
import os
import sqlite3
import sys
import time
from pathlib import Path


PROJECTS_ROOT = Path(os.path.expanduser("~/.finetune-studio/projects"))
DB_CANDIDATES = [
    Path("/home/genortg/finetune-studio/data/finetune_studio.db"),
    Path(os.path.expanduser("~/.finetune-studio/finetune_studio.db")),
    Path(os.path.expanduser("~/finetune_studio/data/finetune_studio.db")),
]


def find_db() -> Path | None:
    for p in DB_CANDIDATES:
        if p.exists():
            return p
    return None


def migrate_project(pid: str, db_path: Path) -> tuple[int, int, int]:
    """Index all filesystem files for a project into project_files.

    Returns (inserted, skipped, errors).
    """
    files_dir = PROJECTS_ROOT / pid / "files"
    if not files_dir.exists():
        print(f"  [{pid}] no files dir at {files_dir}")
        return (0, 0, 0)

    inserted = 0
    skipped = 0
    errors = 0

    with sqlite3.connect(db_path) as conn:
        for hash_dir in sorted(files_dir.iterdir()):
            if not hash_dir.is_dir() or len(hash_dir.name) < 16:
                continue

            metadata_path = hash_dir / "metadata.json"
            if not metadata_path.exists():
                errors += 1
                continue

            try:
                meta = json.loads(metadata_path.read_text(encoding="utf-8"))
            except Exception as e:
                print(f"  [{pid}/{hash_dir.name}] metadata.json parse failed: {e}")
                errors += 1
                continue

            # Use the 12-char dir name as the file id (matches project_files.id pattern)
            file_id = hash_dir.name[:12]
            original_name = meta.get("original_filename") or meta.get("filename") or f"{hash_dir.name}.bin"
            mime_type = meta.get("mime_type") or "application/octet-stream"
            size_bytes = int(meta.get("byte_count") or 0)

            # uploaded_at: prefer metadata, else earliest mtime in dir
            uploaded_at = meta.get("uploaded_at")
            if not uploaded_at:
                try:
                    uploaded_at = min(
                        (p.stat().st_mtime for p in hash_dir.iterdir() if p.is_file()),
                        default=hash_dir.stat().st_mtime,
                    )
                except Exception:
                    uploaded_at = hash_dir.stat().st_mtime

            # uploaded_by: prefer metadata, default 'user'
            uploaded_by = meta.get("uploaded_by") or "user"

            try:
                cur = conn.execute(
                    "INSERT OR IGNORE INTO project_files "
                    "(id, project_id, original_name, mime_type, current_version, size_bytes, uploaded_at, uploaded_by) "
                    "VALUES (?, ?, ?, ?, 1, ?, ?, ?)",
                    (file_id, pid, original_name, mime_type, size_bytes, uploaded_at, uploaded_by),
                )
                if cur.rowcount > 0:
                    inserted += 1
                    print(f"  [{pid}] + {original_name} ({size_bytes} bytes)")
                else:
                    skipped += 1
            except Exception as e:
                print(f"  [{pid}/{original_name}] insert failed: {e}")
                errors += 1
        conn.commit()

    return (inserted, skipped, errors)


def main():
    db_path = find_db()
    if not db_path:
        print(f"ERROR: could not find finetune_studio.db in {DB_CANDIDATES}")
        sys.exit(1)
    print(f"Using DB: {db_path}")
    print(f"Projects root: {PROJECTS_ROOT}")

    pid = sys.argv[1] if len(sys.argv) > 1 else None
    total_i = total_s = total_e = 0

    if pid:
        i, s, e = migrate_project(pid, db_path)
        total_i, total_s, total_e = i, s, e
    else:
        if not PROJECTS_ROOT.exists():
            print(f"projects root not found: {PROJECTS_ROOT}")
            sys.exit(1)
        for proj_dir in sorted(PROJECTS_ROOT.iterdir()):
            if not proj_dir.is_dir():
                continue
            files_dir = proj_dir / "files"
            if not files_dir.exists():
                continue
            i, s, e = migrate_project(proj_dir.name, db_path)
            total_i += i
            total_s += s
            total_e += e

    print(f"\nDONE: inserted={total_i} skipped={total_s} errors={total_e}")


if __name__ == "__main__":
    main()
