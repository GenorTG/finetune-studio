"""File library — disk layout + DB helpers for the per-project file library.

Stage 1 of the REFACTOR-SPEC.

Design:
- Raw files live under {FTS_ROOT}/projects/{pid}/files/raw/{auto_kind}/{id}_{stem}.{ext}
  where {auto_kind} is one of pdfs, imgs, csvs, docs, code, other (MIME-segregated,
  immutable, system-managed).
- Converted files live under {FTS_ROOT}/projects/{pid}/files/converted/{user_folder}/
  and are user-organised. Naming pattern preserves the source stem.
- Folders are DB entities (file_folders table), referenced by name. Raw subfolders
  are also DB rows with kind='auto' so the UI can render them consistently.
- Soft delete: deleted_at + trash_kind fields; on disk the file moves to
  .RAW_TRASH/ or .CONVERTED_TRASH/ under the same root. Restorable until the
  purge job sweeps it (default 7 days).
- All disk paths are absolute (resolved against FTS_ROOT via paths.project_files_root).
"""
from __future__ import annotations

import hashlib
import logging
import mimetypes
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from fastapi import HTTPException
from finetune_studio.data.fs.paths import project_files_root

log = logging.getLogger(__name__)

# ── MIME → auto-kind routing ──────────────────────────────────────────────
# Used to decide which raw subfolder a new upload lands in. The six buckets
# cover 99% of what people feed to a data-prep pipeline; anything else
# goes to 'other'.

_PDF_MIMES = {"application/pdf"}
_IMG_MIMES = {"image/png", "image/jpeg", "image/jpg", "image/gif", "image/webp", "image/tiff", "image/bmp"}
_CSV_MIMES = {"text/csv", "application/csv", "application/vnd.ms-excel"}
_DOC_MIMES = {
    "application/msword",
    "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
    "application/vnd.ms-powerpoint",
    "application/vnd.openxmlformats-officedocument.presentationml.presentation",
    "application/vnd.ms-excel",
    "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
    "application/rtf",
    "text/rtf",
    "application/epub+zip",
    "application/vnd.oasis.opendocument.text",
}
_CODE_MIMES = {"text/x-python", "text/x-java", "text/x-c", "text/x-cpp", "text/javascript", "application/json", "text/html", "text/xml", "text/css", "application/x-sh"}

_AUTO_KINDS = ("pdfs", "imgs", "csvs", "docs", "code", "other")

_RAW_TRASH_DIR = ".RAW_TRASH"
_CONVERTED_TRASH_DIR = ".CONVERTED_TRASH"


def _sniff_mime(filename: str, sniffed: Optional[str] = None) -> str:
    """Best-effort MIME detection: prefer the python-magic 'sniffed' value
    if the caller already ran libmagic, else fall back to mimetypes by
    extension."""
    if sniffed:
        return sniffed
    if not filename:
        return "application/octet-stream"
    mt, _ = mimetypes.guess_type(filename)
    return mt or "application/octet-stream"


def auto_kind_for(mime: str) -> str:
    """Map a MIME type to one of the six raw subfolders."""
    if mime in _PDF_MIMES:
        return "pdfs"
    if mime in _IMG_MIMES:
        return "imgs"
    if mime in _CSV_MIMES:
        return "csvs"
    if mime in _DOC_MIMES:
        return "docs"
    if mime in _CODE_MIMES:
        return "code"
    return "other"


def _ext_for_filename(name: str) -> str:
    """Return lowercase extension WITHOUT the dot, or '' if none."""
    if "." not in name:
        return ""
    return name.rsplit(".", 1)[-1].lower()


def _safe_stem(name: str) -> str:
    """Stem suitable for embedding in a filename. Strips directories, replaces
    any non-[A-Za-z0-9._-] with '_', collapses repeats, caps length."""
    base = name.rsplit("/", 1)[-1].rsplit("\\", 1)[-1]
    stem = base.rsplit(".", 1)[0] if "." in base else base
    if not stem:
        stem = "file"
    cleaned = []
    for ch in stem:
        if ch.isalnum() or ch in "._-":
            cleaned.append(ch)
        else:
            cleaned.append("_")
    s = "".join(cleaned)
    while "__" in s:
        s = s.replace("__", "_")
    return s[:80] or "file"


def sha256_of_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


@dataclass
class FileMetadata:
    file_id: str                # first 16 hex of sha256
    original_name: str
    mime_type: str
    size_bytes: int
    raw_path: str
    raw_hash: str
    auto_kind: str
    version: int

    @property
    def raw_filename(self) -> str:
        return Path(self.raw_path).name


@dataclass
class UploadReportItem:
    """One row in the bulk-upload report."""
    filename: str
    status: str                 # 'uploaded' | 'duplicate' | 'error'
    file_id: Optional[str] = None
    duplicate_of: Optional[str] = None      # filename the duplicate matched
    converted: Optional[str] = None        # 'ok' | 'error' | None (skipped)
    error: Optional[str] = None

    def to_dict(self) -> dict:
        return {
            "filename": self.filename,
            "status": self.status,
            "file_id": self.file_id,
            "duplicate_of": self.duplicate_of,
            "converted": self.converted,
            "error": self.error,
        }


# ── Path helpers ──────────────────────────────────────────────────────────

def raw_subdir(pid: str, auto_kind: str) -> Path:
    """The directory where raw files of this MIME kind live for this project."""
    if auto_kind not in _AUTO_KINDS:
        raise ValueError(f"unknown auto_kind: {auto_kind}")
    return project_files_root(pid) / "raw" / auto_kind


def raw_trash_dir(pid: str) -> Path:
    return project_files_root(pid) / "raw" / _RAW_TRASH_DIR


def converted_trash_dir(pid: str) -> Path:
    return project_files_root(pid) / "converted" / _CONVERTED_TRASH_DIR


def raw_path_for(pid: str, file_id: str, original_name: str, auto_kind: str) -> Path:
    """Build the on-disk path for a raw file. Includes file_id prefix so
    duplicates with the same name don't collide."""
    ext = _ext_for_filename(original_name)
    stem = _safe_stem(original_name)
    return raw_subdir(pid, auto_kind) / f"{file_id}_{stem}{('.' + ext) if ext else ''}"


def converted_filename_for(original_name: str, when_ts: float) -> str:
    """Naming convention for converted files. Includes date for traceability
    when a CLI user wants to find the latest conversion."""
    from datetime import datetime
    stem = _safe_stem(original_name)
    date_str = datetime.fromtimestamp(when_ts).strftime("%Y-%m-%d")
    return f"{stem} (converted {date_str}).md"


def converted_path_for(pid: str, user_folder: str, converted_filename: str) -> Path:
    """Path for a converted file inside a user-named folder."""
    safe_folder = _safe_stem(user_folder) or "default"
    return project_files_root(pid) / "converted" / safe_folder / converted_filename


def ensure_dirs(pid: str) -> None:
    """Create the full directory tree (idempotent)."""
    root = project_files_root(pid)
    for kind in _AUTO_KINDS:
        (root / "raw" / kind).mkdir(parents=True, exist_ok=True)
    (root / "raw" / _RAW_TRASH_DIR).mkdir(parents=True, exist_ok=True)
    (root / "converted").mkdir(parents=True, exist_ok=True)
    (root / "converted" / _CONVERTED_TRASH_DIR).mkdir(parents=True, exist_ok=True)


# ── DB row helpers ────────────────────────────────────────────────────────

def get_or_create_auto_folder(pid: str, auto_kind: str) -> dict:
    """Get the DB row for a raw auto-folder, creating it if missing.

    Auto-folders are the MIME-routed raw subfolders. They're DB rows with
    kind='auto' so the UI treats them like any other folder (read-only).
    """
    from finetune_studio import db
    with db.cursor() as c:
        row = c.execute(
            "SELECT id, project_id, name, kind, parent_id, created_at FROM file_folders WHERE project_id = ? AND name = ? AND kind = 'auto'",
            (pid, auto_kind),
        ).fetchone()
        if row:
            return dict(row)
        fid = f"fld_auto_{auto_kind}_{int(time.time()*1000)}"
        c.execute(
            "INSERT INTO file_folders (id, project_id, name, kind, parent_id, created_at) VALUES (?, ?, ?, 'auto', NULL, ?)",
            (fid, pid, auto_kind, time.time()),
        )
        row = c.execute(
            "SELECT id, project_id, name, kind, parent_id, created_at FROM file_folders WHERE id = ?",
            (fid,),
        ).fetchone()
        return dict(row)


def find_existing_hash(pid: str, raw_hash: str) -> Optional[dict]:
    """Look up an existing live file (deleted_at IS NULL) by its raw hash.
    Used for sha256 dedup on upload."""
    from finetune_studio import db
    with db.cursor() as c:
        row = c.execute(
            """SELECT pf.id, pf.original_name, pf.size_bytes, pf.current_version
                 FROM project_files pf
                 JOIN file_versions fv ON fv.file_id = pf.id AND fv.version = pf.current_version
                WHERE pf.project_id = ? AND fv.raw_hash = ? AND pf.deleted_at IS NULL
                LIMIT 1""",
            (pid, raw_hash),
        ).fetchone()
        return dict(row) if row else None


def record_uploaded_file(
    *,
    pid: str,
    file_id: str,
    original_name: str,
    mime_type: str,
    size_bytes: int,
    raw_path: str,
    raw_hash: str,
    raw_size: int,
    auto_kind: str,
    auto_folder_id: str,
    uploaded_by: str = "user",
) -> dict:
    """Insert a new project_files row + first file_versions row, and add the
    file to the auto-folder. Returns the new row."""
    from finetune_studio import db
    now = time.time()
    with db.cursor() as c:
        c.execute(
            """INSERT INTO project_files
                 (id, project_id, original_name, mime_type, current_version, size_bytes, uploaded_at, uploaded_by)
               VALUES (?, ?, ?, ?, 1, ?, ?, ?)""",
            (file_id, pid, original_name, mime_type, size_bytes, now, uploaded_by),
        )
        c.execute(
            """INSERT INTO file_versions
                 (file_id, version, raw_path, raw_hash, raw_size, uploaded_at, uploaded_by)
               VALUES (?, 1, ?, ?, ?, ?, ?)""",
            (file_id, raw_path, raw_hash, raw_size, now, uploaded_by),
        )
        c.execute(
            "INSERT OR IGNORE INTO folder_membership (folder_id, file_id) VALUES (?, ?)",
            (auto_folder_id, file_id),
        )
        row = c.execute(
            "SELECT id, project_id, original_name, mime_type, current_version, size_bytes, uploaded_at, uploaded_by FROM project_files WHERE id = ?",
            (file_id,),
        ).fetchone()
    return dict(row)


def list_files(
    pid: str,
    *,
    folder_id: Optional[str] = None,
    include_deleted: bool = False,
    mime_prefix: Optional[str] = None,
    search: Optional[str] = None,
) -> list[dict]:
    """List files for a project. If folder_id is given, restrict to files
    in that folder. Otherwise return all live files."""
    from finetune_studio import db
    clauses = ["pf.project_id = ?"]
    params: list = [pid]
    if not include_deleted:
        clauses.append("pf.deleted_at IS NULL")
    if folder_id:
        clauses.append("pf.id IN (SELECT file_id FROM folder_membership WHERE folder_id = ?)")
        params.append(folder_id)
    if mime_prefix:
        clauses.append("pf.mime_type LIKE ?")
        params.append(mime_prefix + "%")
    if search:
        clauses.append("pf.original_name LIKE ?")
        params.append("%" + search + "%")
    sql = f"""SELECT pf.id, pf.original_name, pf.mime_type, pf.current_version,
                     pf.size_bytes, pf.uploaded_at, pf.uploaded_by,
                     pf.last_trained_at, pf.deleted_at, pf.tags, pf.notes,
                     (SELECT name FROM file_folders WHERE id IN
                        (SELECT folder_id FROM folder_membership WHERE file_id = pf.id)) AS first_folder,
                     (SELECT folder_id FROM folder_membership WHERE file_id = pf.id LIMIT 1) AS folder_id
                FROM project_files pf
                WHERE {' AND '.join(clauses)}
                ORDER BY pf.uploaded_at DESC"""
    with db.cursor() as c:
        rows = c.execute(sql, params).fetchall()
    return [dict(r) for r in rows]


def list_folders(pid: str, *, include_auto: bool = True) -> list[dict]:
    from finetune_studio import db
    with db.cursor() as c:
        if include_auto:
            rows = c.execute(
                "SELECT id, project_id, name, kind, parent_id, created_at FROM file_folders WHERE project_id = ? ORDER BY kind, name",
                (pid,),
            ).fetchall()
        else:
            rows = c.execute(
                "SELECT id, project_id, name, kind, parent_id, created_at FROM file_folders WHERE project_id = ? AND kind = 'user' ORDER BY name",
                (pid,),
            ).fetchall()
    return [dict(r) for r in rows]


def get_file(pid: str, file_id: str, *, include_deleted: bool = False) -> Optional[dict]:
    from finetune_studio import db
    clauses = ["pf.project_id = ?", "pf.id = ?"]
    params: list = [pid, file_id]
    if not include_deleted:
        clauses.append("pf.deleted_at IS NULL")
    sql = f"""SELECT pf.id, pf.project_id, pf.original_name, pf.mime_type,
                     pf.current_version, pf.size_bytes, pf.uploaded_at,
                     pf.uploaded_by, pf.last_trained_at, pf.deleted_at,
                     pf.tags, pf.notes
                FROM project_files pf
                WHERE {' AND '.join(clauses)}
                LIMIT 1"""
    with db.cursor() as c:
        row = c.execute(sql, params).fetchone()
    return dict(row) if row else None


def list_versions(pid: str, file_id: str) -> list[dict]:
    """Return all raw versions of a file, newest first."""
    from finetune_studio import db
    with db.cursor() as c:
        rows = c.execute(
            """SELECT id, file_id, version, raw_path, raw_hash, raw_size, uploaded_at, uploaded_by
                 FROM file_versions
                WHERE file_id = ?
                ORDER BY version DESC""",
            (file_id,),
        ).fetchall()
    return [dict(r) for r in rows]


def list_conversions(pid: str, file_id: str) -> list[dict]:
    from finetune_studio import db
    with db.cursor() as c:
        rows = c.execute(
            """SELECT id, file_id, version, format, converted_path, converted_hash,
                     converted_size, converter, converted_at, status, error_message
                 FROM file_conversions
                WHERE file_id = ?
                ORDER BY converted_at DESC""",
            (file_id,),
        ).fetchall()
    return [dict(r) for r in rows]


def create_folder(pid: str, name: str) -> dict:
    from finetune_studio import db
    safe = _safe_stem(name)
    if not safe:
        raise HTTPException(status_code=400, detail="invalid folder name")
    fid = f"fld_user_{int(time.time()*1000)}_{safe}"
    now = time.time()
    try:
        with db.cursor() as c:
            c.execute(
                "INSERT INTO file_folders (id, project_id, name, kind, parent_id, created_at) VALUES (?, ?, ?, 'user', NULL, ?)",
                (fid, pid, safe, now),
            )
            # Also create the on-disk dir under files/converted/{name}/
            (project_files_root(pid) / "converted" / safe).mkdir(parents=True, exist_ok=True)
            row = c.execute(
                "SELECT id, project_id, name, kind, parent_id, created_at FROM file_folders WHERE id = ?",
                (fid,),
            ).fetchone()
    except Exception as e:
        # SQLite UNIQUE constraint
        if "UNIQUE" in str(e):
            raise HTTPException(status_code=409, detail=f"folder '{safe}' already exists")
        raise
    return dict(row)


def rename_folder(pid: str, folder_id: str, new_name: str) -> dict:
    from finetune_studio import db
    safe = _safe_stem(new_name)
    if not safe:
        raise HTTPException(status_code=400, detail="invalid folder name")
    with db.cursor() as c:
        row = c.execute(
            "SELECT id, kind, name FROM file_folders WHERE id = ? AND project_id = ?",
            (folder_id, pid),
        ).fetchone()
        if not row:
            raise HTTPException(status_code=404, detail="folder not found")
        if row["kind"] == "auto":
            raise HTTPException(status_code=403, detail="cannot rename auto-folder")
        try:
            c.execute(
                "UPDATE file_folders SET name = ? WHERE id = ?",
                (safe, folder_id),
            )
        except Exception as e:
            if "UNIQUE" in str(e):
                raise HTTPException(status_code=409, detail=f"folder '{safe}' already exists")
            raise
        # Best-effort disk rename (only if the dir exists and is empty of files
        # at root; converted files live inside it so we just rename the dir).
        old_name = row["name"]
        old_dir = project_files_root(pid) / "converted" / old_name
        new_dir = project_files_root(pid) / "converted" / safe
        if old_dir.exists() and not new_dir.exists():
            try:
                old_dir.rename(new_dir)
            except OSError:
                pass
        updated = c.execute(
            "SELECT id, project_id, name, kind, parent_id, created_at FROM file_folders WHERE id = ?",
            (folder_id,),
        ).fetchone()
    return dict(updated)


def delete_folder(pid: str, folder_id: str) -> dict:
    """Delete a user folder. The files in it remain — only the membership
    rows are removed (so files become folder-less, ready to be re-bucketed)."""
    from finetune_studio import db
    with db.cursor() as c:
        row = c.execute(
            "SELECT id, kind FROM file_folders WHERE id = ? AND project_id = ?",
            (folder_id, pid),
        ).fetchone()
        if not row:
            raise HTTPException(status_code=404, detail="folder not found")
        if row["kind"] == "auto":
            raise HTTPException(status_code=403, detail="cannot delete auto-folder")
        c.execute("DELETE FROM folder_membership WHERE folder_id = ?", (folder_id,))
        c.execute("DELETE FROM file_folders WHERE id = ?", (folder_id,))
    return {"deleted": folder_id}


def move_file_to_folder(pid: str, file_id: str, folder_id: str) -> dict:
    """Move a file to a different folder.

    Invariants (enforced here so the UI cannot violate them):
    - Files currently in an auto (raw MIME) folder are pinned: they cannot
      be moved to a user folder. Raw bytes live at their content-addressed
      MIME path forever so the audit trail is stable.
    - Files in an auto folder can only move to the SAME-MIME auto folder
      (which is a no-op, but reject explicitly to catch logic errors).
    - Files in a user folder can move to any other user folder freely.
    """
    from finetune_studio import db
    with db.cursor() as c:
        f = c.execute(
            "SELECT id, mime_type FROM project_files WHERE id = ? AND project_id = ? AND deleted_at IS NULL",
            (file_id, pid),
        ).fetchone()
        if not f:
            raise HTTPException(status_code=404, detail="file not found")
        target = c.execute(
            "SELECT id, kind, name FROM file_folders WHERE id = ? AND project_id = ?",
            (folder_id, pid),
        ).fetchone()
        if not target:
            raise HTTPException(status_code=404, detail="folder not found")

        # Find the file's current folder
        current = c.execute(
            """SELECT ff.id, ff.kind, ff.name FROM file_folders ff
                 JOIN folder_membership fm ON fm.folder_id = ff.id
                WHERE fm.file_id = ?""",
            (file_id,),
        ).fetchone()

        # Rule 1: a file currently in an auto folder is pinned to it.
        if current and current["kind"] == "auto":
            if target["kind"] == "user":
                raise HTTPException(
                    status_code=409,
                    detail=(
                        f"raw files are pinned to {current['name']}/ and cannot be moved to "
                        f"user folders; only the converted version of this file can be moved"
                    ),
                )
            # target is auto — only allow if same MIME kind
            file_kind = auto_kind_for(f["mime_type"])
            if file_kind != target["name"]:
                raise HTTPException(
                    status_code=409,
                    detail=f"file mime {f['mime_type']} does not belong in {target['name']}/",
                )

        # Rule 2: a file in a user folder cannot be moved to an auto folder.
        # Auto folders are MIME-routed system folders; only the upload pipeline
        # writes into them.
        if current and current["kind"] == "user" and target["kind"] == "auto":
            raise HTTPException(
                status_code=409,
                detail="user-organised files cannot be moved into auto (raw) folders",
            )

        # If we get here, the move is legal. Apply it.
        c.execute("DELETE FROM folder_membership WHERE file_id = ?", (file_id,))
        c.execute(
            "INSERT OR IGNORE INTO folder_membership (folder_id, file_id) VALUES (?, ?)",
            (folder_id, file_id),
        )
    return {"file_id": file_id, "folder_id": folder_id}


def soft_delete_file(pid: str, file_id: str) -> dict:
    """Soft-delete: move the file (and any converted siblings) to the trash
    dirs and set deleted_at. Reversible via restore."""
    from finetune_studio import db
    ensure_dirs(pid)
    now = time.time()
    with db.cursor() as c:
        f = c.execute(
            "SELECT id, original_name FROM project_files WHERE id = ? AND project_id = ? AND deleted_at IS NULL",
            (file_id, pid),
        ).fetchone()
        if not f:
            raise HTTPException(status_code=404, detail="file not found or already deleted")
        # Move raw versions
        versions = c.execute(
            "SELECT id, raw_path FROM file_versions WHERE file_id = ?", (file_id,)
        ).fetchall()
        for v in versions:
            src = Path(v["raw_path"])
            if src.exists():
                dst = raw_trash_dir(pid) / src.name
                try:
                    src.rename(dst)
                except OSError:
                    pass
        # Move converted siblings
        convs = c.execute(
            "SELECT converted_path FROM file_conversions WHERE file_id = ?", (file_id,)
        ).fetchall()
        for cv in convs:
            src = Path(cv["converted_path"])
            if src.exists():
                dst = converted_trash_dir(pid) / src.name
                try:
                    src.rename(dst)
                except OSError:
                    pass
        c.execute(
            "UPDATE project_files SET deleted_at = ?, trash_kind = 'raw' WHERE id = ?",
            (now, file_id),
        )
    return {"deleted": file_id, "deleted_at": now}


def restore_file(pid: str, file_id: str) -> dict:
    """Restore a soft-deleted file from trash back to its original auto-folder.
    Converted siblings stay in trash — they'd need to be re-converted."""
    from finetune_studio import db
    now = time.time()
    with db.cursor() as c:
        f = c.execute(
            "SELECT id, original_name, mime_type, current_version FROM project_files WHERE id = ? AND project_id = ? AND deleted_at IS NOT NULL",
            (file_id, pid),
        ).fetchone()
        if not f:
            raise HTTPException(status_code=404, detail="file not in trash")
        # Find the live (non-trash) raw on disk for the current version
        live_version = c.execute(
            "SELECT raw_path FROM file_versions WHERE file_id = ? AND version = ?",
            (file_id, f["current_version"]),
        ).fetchone()
        if live_version:
            src = Path(live_version["raw_path"])
            if not src.exists():
                # The file is in trash; move it back to its original MIME folder
                candidate = None
                for cand in raw_trash_dir(pid).glob(f"{file_id}_*"):
                    candidate = cand
                    break
                if candidate is None:
                    raise HTTPException(status_code=409, detail="trash file missing on disk")
                kind = auto_kind_for(f["mime_type"])
                target_dir = raw_subdir(pid, kind)
                target_dir.mkdir(parents=True, exist_ok=True)
                target_path = target_dir / candidate.name
                try:
                    candidate.rename(target_path)
                except OSError as e:
                    raise HTTPException(status_code=500, detail=f"restore failed: {e}")
                c.execute(
                    "UPDATE file_versions SET raw_path = ? WHERE file_id = ? AND version = ?",
                    (str(target_path), file_id, f["current_version"]),
                )
        # Clear deleted_at
        c.execute(
            "UPDATE project_files SET deleted_at = NULL, trash_kind = NULL WHERE id = ?",
            (file_id,),
        )
    return {"restored": file_id}


def purge_trash(pid: str, older_than_days: int = 7) -> dict:
    """Hard-delete everything in trash older than N days."""
    from finetune_studio import db
    cutoff = time.time() - older_than_days * 86400
    purged_files: list[str] = []
    with db.cursor() as c:
        rows = c.execute(
            "SELECT id, original_name, deleted_at FROM project_files WHERE project_id = ? AND deleted_at IS NOT NULL AND deleted_at < ?",
            (pid, cutoff),
        ).fetchall()
        for r in rows:
            fid = r["id"]
            # Remove files from disk
            versions = c.execute(
                "SELECT raw_path FROM file_versions WHERE file_id = ?", (fid,)
            ).fetchall()
            for v in versions:
                p = Path(v["raw_path"])
                if p.exists() and _RAW_TRASH_DIR in p.parts:
                    try:
                        p.unlink()
                    except OSError:
                        pass
            convs = c.execute(
                "SELECT converted_path FROM file_conversions WHERE file_id = ?", (fid,)
            ).fetchall()
            for cv in convs:
                p = Path(cv["converted_path"])
                if p.exists() and _CONVERTED_TRASH_DIR in p.parts:
                    try:
                        p.unlink()
                    except OSError:
                        pass
            # Cascade DB rows
            c.execute("DELETE FROM file_conversions WHERE file_id = ?", (fid,))
            c.execute("DELETE FROM file_versions WHERE file_id = ?", (fid,))
            c.execute("DELETE FROM folder_membership WHERE file_id = ?", (fid,))
            c.execute("DELETE FROM project_files WHERE id = ?", (fid,))
            purged_files.append(r["original_name"])
    return {"purged": purged_files, "count": len(purged_files)}


def list_trash(pid: str) -> list[dict]:
    """List files currently in trash, with how many days until purge."""
    from finetune_studio import db
    now = time.time()
    with db.cursor() as c:
        rows = c.execute(
            """SELECT id, original_name, mime_type, deleted_at, trash_kind
                 FROM project_files
                WHERE project_id = ? AND deleted_at IS NOT NULL
                ORDER BY deleted_at DESC""",
            (pid,),
        ).fetchall()
    out = []
    for r in rows:
        d = dict(r)
        d["age_days"] = round((now - r["deleted_at"]) / 86400, 1)
        out.append(d)
    return out


# ── Single-file write path (used by routes/file_library.py) ──────────────

def write_uploaded_file(pid: str, data: bytes, original_name: str, *, mime_hint: Optional[str] = None, uploaded_by: str = "user") -> FileMetadata:
    """Write bytes to the correct MIME-segregated raw subfolder. Compute
    sha256. Create the project_files + file_versions rows. Returns metadata.

    Dedup is NOT applied here — the caller (route handler) checks for an
    existing hash first and decides whether to call this.
    """
    ensure_dirs(pid)
    mime = _sniff_mime(original_name, mime_hint)
    kind = auto_kind_for(mime)
    raw_hash = sha256_of_bytes(data)
    file_id = raw_hash[:16]
    auto_folder = get_or_create_auto_folder(pid, kind)

    raw_path = raw_path_for(pid, file_id, original_name, kind)
    raw_path.parent.mkdir(parents=True, exist_ok=True)
    raw_path.write_bytes(data)

    row = record_uploaded_file(
        pid=pid,
        file_id=file_id,
        original_name=original_name,
        mime_type=mime,
        size_bytes=len(data),
        raw_path=str(raw_path),
        raw_hash=raw_hash,
        raw_size=len(data),
        auto_kind=kind,
        auto_folder_id=auto_folder["id"],
        uploaded_by=uploaded_by,
    )
    return FileMetadata(
        file_id=file_id,
        original_name=original_name,
        mime_type=mime,
        size_bytes=len(data),
        raw_path=str(raw_path),
        raw_hash=raw_hash,
        auto_kind=kind,
        version=row["current_version"],
    )


# ── Rename / per-file purge / parsed MD ───────────────────────────────────

# Process-lifetime cache for GET .../parsed. Keyed by "pid:fid".
_PARSED_CACHE: dict[str, dict] = {}

_BINARY_EXTS = frozenset({
    "pdf", "docx", "doc", "pptx", "ppt", "xlsx", "xls",
    "png", "jpg", "jpeg", "gif", "webp", "bmp", "tiff", "tif",
    "zip", "gz", "bz2", "7z", "rar", "bin", "exe", "dll",
    "woff", "woff2", "ttf", "otf", "ico",
})
_CONVERTIBLE_EXTS = frozenset({"txt", "csv", "json", "md", "markdown", "tsv", "log"})


def _cache_key(pid: str, file_id: str) -> str:
    return f"{pid}:{file_id}"


def invalidate_parsed_cache(pid: str, file_id: str) -> None:
    """Drop any in-memory parsed-MD cache entry for this file."""
    _PARSED_CACHE.pop(_cache_key(pid, file_id), None)


def _project_files_columns() -> set[str]:
    """Return the live column names on project_files (for optional parsed_* cols)."""
    from finetune_studio import db
    with db.cursor() as c:
        rows = c.execute("PRAGMA table_info(project_files)").fetchall()
    names: set[str] = set()
    for r in rows:
        # sqlite3.Row supports both index and name access
        try:
            names.add(str(r["name"]))
        except (KeyError, IndexError, TypeError):
            names.add(str(r[1]))
    return names


def _validate_rename_name(new_name: str) -> str:
    """Validate a user-facing rename target. Returns the stripped name.

    Rejects empty names, path separators, and unsafe (non-alnum) extensions.
    """
    name = (new_name or "").strip()
    if not name:
        raise HTTPException(
            status_code=400, detail="new_name must be non-empty"
        )
    if "/" in name or "\\" in name or name in {".", ".."} or ".." in name:
        raise HTTPException(
            status_code=400,
            detail="path separators not allowed in new_name",
        )
    # Basename only — reject absolute / drive-like names
    if os.path.basename(name) != name:
        raise HTTPException(
            status_code=400,
            detail="path separators not allowed in new_name",
        )
    ext = _ext_for_filename(name)
    if ext and not ext.replace("_", "").isalnum():
        raise HTTPException(
            status_code=400, detail=f"unsafe extension: .{ext}"
        )
    return name


def _csv_to_md_table(text: str) -> str:
    """Convert CSV/TSV text to a simple GitHub-flavoured markdown table."""
    import csv
    from io import StringIO

    sample = text[:4096]
    try:
        dialect = csv.Sniffer().sniff(sample, delimiters=",\t;")
    except csv.Error:
        dialect = csv.excel
    reader = csv.reader(StringIO(text), dialect)
    rows = [list(r) for r in reader]
    if not rows:
        return ""
    width = max(len(r) for r in rows)
    normalized: list[list[str]] = []
    for r in rows:
        cells = [c.replace("|", "\\|").replace("\n", " ") for c in r]
        if len(cells) < width:
            cells.extend([""] * (width - len(cells)))
        normalized.append(cells[:width])
    header = normalized[0]
    lines = [
        "| " + " | ".join(header) + " |",
        "| " + " | ".join("---" for _ in header) + " |",
    ]
    for row in normalized[1:]:
        lines.append("| " + " | ".join(row) + " |")
    return "\n".join(lines) + "\n"


def _convert_raw_to_md(path: Path, original_name: str) -> str:
    """Convert a text-like file on disk into markdown. Raises HTTPException 422
    for binary / unsupported formats."""
    ext = _ext_for_filename(original_name)
    if ext in _BINARY_EXTS:
        raise HTTPException(
            status_code=422,
            detail=(
                f"cannot convert binary format '.{ext}' to markdown inline — "
                "run Data Prep → Prep job (or upload a sibling .md) first"
            ),
        )
    if ext not in _CONVERTIBLE_EXTS and ext not in {"", "text"}:
        # Unknown extension: try UTF-8 read; if it looks binary, 422
        raw = path.read_bytes()
        if b"\x00" in raw[:8192]:
            raise HTTPException(
                status_code=422,
                detail=(
                    f"file looks binary (extension '.{ext or '?'}'); "
                    "cannot convert to markdown"
                ),
            )
        text = raw.decode("utf-8", errors="replace")
        return f"```\n{text}\n```\n"

    text = path.read_text(encoding="utf-8", errors="replace")
    if ext in {"md", "markdown"}:
        return text
    if ext == "json":
        return f"```json\n{text.strip()}\n```\n"
    if ext in {"csv", "tsv"}:
        return _csv_to_md_table(text)
    # txt / log / plain
    return f"```\n{text}\n```\n"


def _current_raw_path(pid: str, file_id: str, current_version: int) -> Optional[Path]:
    """Resolve the on-disk path for the file's current version.

    Soft-delete moves bytes into .RAW_TRASH without always updating raw_path,
    so we fall back to a trash glob when the recorded path is missing.
    """
    from finetune_studio import db
    with db.cursor() as c:
        row = c.execute(
            "SELECT raw_path FROM file_versions WHERE file_id = ? AND version = ?",
            (file_id, current_version),
        ).fetchone()
    if not row:
        return None
    p = Path(row["raw_path"])
    if p.exists():
        return p
    # Soft-deleted: look in trash for {file_id}_*
    trash = raw_trash_dir(pid)
    if trash.exists():
        for cand in trash.glob(f"{file_id}_*"):
            if cand.is_file():
                return cand
    return p if p.exists() else None


def rename_file(pid: str, file_id: str, new_name: str) -> dict:
    """Rename a live file: update project_files.original_name and rename on disk.

    Returns ``{ok: True, file: {...}}``. Raises 404 / 409 / 400 via HTTPException.
    """
    from finetune_studio import db

    safe_name = _validate_rename_name(new_name)
    with db.cursor() as c:
        f = c.execute(
            """SELECT id, original_name, mime_type, current_version, size_bytes,
                      uploaded_at, uploaded_by, deleted_at, tags, notes
                 FROM project_files
                WHERE id = ? AND project_id = ?""",
            (file_id, pid),
        ).fetchone()
        if not f:
            raise HTTPException(status_code=404, detail="file not found")
        if f["deleted_at"] is not None:
            raise HTTPException(
                status_code=409,
                detail="cannot rename a trashed file; restore it first",
            )
        if f["original_name"] == safe_name:
            return {"ok": True, "file": get_file(pid, file_id)}

        clash = c.execute(
            """SELECT id FROM project_files
                WHERE project_id = ? AND original_name = ?
                  AND id != ? AND deleted_at IS NULL""",
            (pid, safe_name, file_id),
        ).fetchone()
        if clash:
            raise HTTPException(
                status_code=409,
                detail=(
                    f"a file named '{safe_name}' already exists "
                    "in this project"
                ),
            )

        ver = c.execute(
            "SELECT id, raw_path FROM file_versions "
            "WHERE file_id = ? AND version = ?",
            (file_id, f["current_version"]),
        ).fetchone()
        if not ver:
            raise HTTPException(
                status_code=404, detail="file version missing"
            )

        old_path = Path(ver["raw_path"])
        kind = auto_kind_for(
            f["mime_type"] or "application/octet-stream"
        )
        # Prefer keeping the file in its current directory (may differ from
        # MIME auto-folder after a move); only rebuild the filename.
        if old_path.exists():
            new_name_on_disk = raw_path_for(
                pid, file_id, safe_name, kind
            ).name
            new_path = old_path.parent / new_name_on_disk
            if new_path != old_path:
                if new_path.exists():
                    raise HTTPException(
                        status_code=409,
                        detail=(
                            "target path already exists on disk: "
                            f"{new_path.name}"
                        ),
                    )
                try:
                    old_path.rename(new_path)
                except OSError as e:
                    raise HTTPException(
                        status_code=500,
                        detail=f"disk rename failed: {e}",
                    ) from e
            c.execute(
                "UPDATE file_versions SET raw_path = ? WHERE id = ?",
                (str(new_path), ver["id"]),
            )
        else:
            log.warning(
                "rename: raw missing on disk for %s (%s)",
                file_id,
                old_path,
            )

        c.execute(
            "UPDATE project_files SET original_name = ? WHERE id = ?",
            (safe_name, file_id),
        )

    invalidate_parsed_cache(pid, file_id)
    updated = get_file(pid, file_id)
    if not updated:
        raise HTTPException(status_code=404, detail="file not found after rename")
    return {"ok": True, "file": updated}


def purge_file(pid: str, file_id: str) -> dict:
    """Hard-delete a single trashed file (disk + DB rows).

    File must already be soft-deleted (``deleted_at IS NOT NULL``). Returns
    ``{ok: True, fid: '...'}``.
    """
    from finetune_studio import db

    with db.cursor() as c:
        f = c.execute(
            """SELECT id, original_name, deleted_at FROM project_files
                WHERE id = ? AND project_id = ?""",
            (file_id, pid),
        ).fetchone()
        if not f:
            raise HTTPException(status_code=404, detail="file not found")
        if f["deleted_at"] is None:
            raise HTTPException(
                status_code=409,
                detail=(
                    "file is not in trash — soft-delete it first "
                    "(DELETE /files/{fid})"
                ),
            )

        versions = c.execute(
            "SELECT raw_path FROM file_versions WHERE file_id = ?", (file_id,)
        ).fetchall()
        for v in versions:
            p = Path(v["raw_path"])
            if p.exists():
                try:
                    p.unlink()
                except OSError:
                    pass
        # Soft-delete may have moved bytes to trash without updating raw_path
        trash = raw_trash_dir(pid)
        if trash.exists():
            for cand in trash.glob(f"{file_id}_*"):
                try:
                    cand.unlink()
                except OSError:
                    pass

        convs = c.execute(
            "SELECT converted_path FROM file_conversions WHERE file_id = ?", (file_id,)
        ).fetchall()
        for cv in convs:
            p = Path(cv["converted_path"])
            if p.exists():
                try:
                    p.unlink()
                except OSError:
                    pass
        conv_trash = converted_trash_dir(pid)
        if conv_trash.exists():
            for cand in conv_trash.iterdir():
                # Best-effort: converted trash names don't always include file_id
                if cand.is_file() and file_id in cand.name:
                    try:
                        cand.unlink()
                    except OSError:
                        pass

        c.execute("DELETE FROM file_conversions WHERE file_id = ?", (file_id,))
        c.execute("DELETE FROM file_versions WHERE file_id = ?", (file_id,))
        c.execute("DELETE FROM folder_membership WHERE file_id = ?", (file_id,))
        c.execute("DELETE FROM project_files WHERE id = ?", (file_id,))

    invalidate_parsed_cache(pid, file_id)
    return {"ok": True, "fid": file_id}


def get_parsed_markdown(pid: str, file_id: str) -> dict:
    """Resolve a file's parsed-markdown representation.

    Resolution order:
      1. Optional ``parsed_md`` / ``parsed_path`` columns on project_files (source=db)
      2. ``file_conversions`` row with format md/txt and status ok (source=db)
      3. Sibling ``.md`` next to the stored raw file (source=sibling)
      4. Legacy ``files/<sha>/parsed.txt`` (source=sibling)
      5. On-the-fly conversion of .txt / .csv / .json / .md (source=converted)
      6. 422 for binary / unsupported formats

    Results are cached in-process for the lifetime of the server.
    """
    from finetune_studio import db

    cache_k = _cache_key(pid, file_id)
    if cache_k in _PARSED_CACHE:
        return _PARSED_CACHE[cache_k]

    f = get_file(pid, file_id, include_deleted=True)
    if not f:
        raise HTTPException(status_code=404, detail="file not found")

    cols = _project_files_columns()
    raw_path_obj = _current_raw_path(pid, file_id, int(f["current_version"] or 1))
    path_str = str(raw_path_obj) if raw_path_obj else ""

    # 1) Optional DB columns (not present in current schema — checked live)
    if "parsed_md" in cols or "parsed_path" in cols:
        with db.cursor() as c:
            row = c.execute(
                "SELECT * FROM project_files WHERE id = ? AND project_id = ?",
                (file_id, pid),
            ).fetchone()
        if row:
            if "parsed_md" in cols and row["parsed_md"]:
                result = {
                    "fid": file_id,
                    "path": path_str,
                    "parsed_md": str(row["parsed_md"]),
                    "source": "db",
                }
                _PARSED_CACHE[cache_k] = result
                return result
            if "parsed_path" in cols and row["parsed_path"]:
                p = Path(str(row["parsed_path"]))
                if p.exists():
                    result = {
                        "fid": file_id,
                        "path": str(p),
                        "parsed_md": p.read_text(encoding="utf-8", errors="replace"),
                        "source": "db",
                    }
                    _PARSED_CACHE[cache_k] = result
                    return result

    # 2) file_conversions table (converted MD/txt written by prep pipeline)
    with db.cursor() as c:
        conv = c.execute(
            """SELECT converted_path, format, status FROM file_conversions
                WHERE file_id = ?
                  AND lower(format) IN ('md', 'txt', 'markdown')
                  AND lower(status) != 'error'
                ORDER BY converted_at DESC LIMIT 1""",
            (file_id,),
        ).fetchone()
    if conv:
        cp = Path(conv["converted_path"])
        if cp.exists():
            result = {
                "fid": file_id,
                "path": str(cp),
                "parsed_md": cp.read_text(encoding="utf-8", errors="replace"),
                "source": "db",
            }
            _PARSED_CACHE[cache_k] = result
            return result

    # 3) Sibling .md next to stored raw
    if raw_path_obj is not None:
        sibling = raw_path_obj.with_suffix(".md")
        if sibling.exists() and sibling.is_file() and sibling != raw_path_obj:
            result = {
                "fid": file_id,
                "path": str(sibling),
                "parsed_md": sibling.read_text(encoding="utf-8", errors="replace"),
                "source": "sibling",
            }
            _PARSED_CACHE[cache_k] = result
            return result
        # Also accept same-stem .md without the file_id_ prefix in the same dir
        stem = _safe_stem(f["original_name"])
        alt = raw_path_obj.parent / f"{stem}.md"
        if alt.exists() and alt.is_file():
            result = {
                "fid": file_id,
                "path": str(alt),
                "parsed_md": alt.read_text(encoding="utf-8", errors="replace"),
                "source": "sibling",
            }
            _PARSED_CACHE[cache_k] = result
            return result

    # 4) Legacy data-prep path: files/<sha12>/parsed.txt (do NOT call file_dir —
    # it mkdir's as a side effect).
    root = project_files_root(pid)
    for short in (file_id[:12], file_id[:16], file_id):
        legacy = root / short / "parsed.txt"
        if legacy.exists() and legacy.is_file():
            result = {
                "fid": file_id,
                "path": str(legacy),
                "parsed_md": legacy.read_text(encoding="utf-8", errors="replace"),
                "source": "sibling",
            }
            _PARSED_CACHE[cache_k] = result
            return result

    # 5) On-the-fly conversion from raw
    if raw_path_obj is None or not raw_path_obj.exists():
        raise HTTPException(status_code=410, detail="file missing on disk")

    md = _convert_raw_to_md(raw_path_obj, f["original_name"])
    result = {
        "fid": file_id,
        "path": str(raw_path_obj),
        "parsed_md": md,
        "source": "converted",
    }
    _PARSED_CACHE[cache_k] = result
    return result
