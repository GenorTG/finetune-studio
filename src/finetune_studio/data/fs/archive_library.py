"""Project-archive helpers: carry the DB-backed file library and re-base paths.

The archive tarball holds the project directory, but the file library (folders,
files, versions, conversions) lives in SQLite and parsed-source manifests
record absolute paths into the *old* project dir. Without these helpers an
imported project lists zero files and every source fails the data-prep path
fence (it points outside the new project directory).
"""
from __future__ import annotations

import re
from pathlib import Path
from typing import Any

LIBRARY_MEMBER = "file_library.json"
_TABLES = ("file_folders", "project_files", "file_versions", "file_conversions", "folder_membership")


def dump_file_library(pid: str) -> dict[str, list[dict[str, Any]]]:
    """Rows of the file-library tables belonging to project ``pid``."""
    from finetune_studio import db

    with db.cursor() as c:
        folders = [dict(r) for r in c.execute("SELECT * FROM file_folders WHERE project_id = ?", (pid,))]
        files = [dict(r) for r in c.execute("SELECT * FROM project_files WHERE project_id = ?", (pid,))]
        marks = ",".join("?" * len(files)) or "''"
        ids = [f["id"] for f in files]
        versions = [dict(r) for r in c.execute(
            f"SELECT * FROM file_versions WHERE file_id IN ({marks})", ids)]
        convs = [dict(r) for r in c.execute(
            f"SELECT * FROM file_conversions WHERE file_id IN ({marks})", ids)]
        member = [dict(r) for r in c.execute(
            f"SELECT * FROM folder_membership WHERE file_id IN ({marks})", ids)]
    return {"file_folders": folders, "project_files": files, "file_versions": versions,
            "file_conversions": convs, "folder_membership": member}


def rebase_text(text: str, old_pid: str, new_dir: Path) -> str:
    """Point every ``.../projects/<old_pid>/`` prefix at ``new_dir``."""
    pat = re.compile(r"[^\"\s]*?/projects/" + re.escape(old_pid) + r"/")
    return pat.sub(lambda _m: str(new_dir).replace("\\", "/") + "/", text)


def rebase_project_files(project_dir: Path, old_pid: str) -> int:
    """Rewrite stale absolute paths in source manifests/logs. Returns files changed."""
    changed = 0
    targets = [*project_dir.glob("qa/sources/*.json"), *project_dir.glob("logs/*.jsonl")]
    for f in targets:
        try:
            before = f.read_text(encoding="utf-8")
        except (OSError, UnicodeDecodeError):
            continue
        after = rebase_text(before, old_pid, project_dir)
        if after != before:
            f.write_text(after, encoding="utf-8")
            changed += 1
    return changed


def restore_file_library(dump: Any, old_pid: str, new_pid: str, new_dir: Path) -> int:
    """Insert ``dump`` under ``new_pid`` with fresh ids and re-based paths.

    Returns the number of file rows restored. Malformed dumps restore nothing.
    """
    from finetune_studio import db

    if not isinstance(dump, dict) or not all(isinstance(dump.get(t, []), list) for t in _TABLES):
        return 0

    def fid(old: str) -> str:
        return old.replace(old_pid, new_pid, 1) if old.startswith(old_pid) else f"{new_pid}-{old}"

    def fold(old: str) -> str:
        return f"{old}-{new_pid}"

    def cols(table: str, row: dict, **over: Any) -> tuple[str, list[Any]]:
        data = {**row, **over}
        if table in ("file_versions", "file_conversions"):
            data.pop("id", None)  # autoincrement
        names = list(data)
        return (f"INSERT OR IGNORE INTO {table} ({','.join(names)}) VALUES ({','.join('?' * len(names))})",
                [data[n] for n in names])

    n_files = 0
    with db.cursor() as c:
        for r in dump.get("file_folders", []):
            parent = r.get("parent_id")
            sql, args = cols("file_folders", r, id=fold(r["id"]), project_id=new_pid,
                             parent_id=fold(parent) if parent else None)
            c.execute(sql, args)
        for r in dump.get("project_files", []):
            sql, args = cols("project_files", r, id=fid(r["id"]), project_id=new_pid)
            c.execute(sql, args)
            n_files += 1
        for r in dump.get("file_versions", []):
            sql, args = cols("file_versions", r, file_id=fid(r["file_id"]),
                             raw_path=rebase_text(r["raw_path"], old_pid, new_dir))
            c.execute(sql, args)
        for r in dump.get("file_conversions", []):
            sql, args = cols("file_conversions", r, file_id=fid(r["file_id"]),
                             converted_path=rebase_text(r["converted_path"], old_pid, new_dir))
            c.execute(sql, args)
        for r in dump.get("folder_membership", []):
            sql, args = cols("folder_membership", r, folder_id=fold(r["folder_id"]),
                             file_id=fid(r["file_id"]))
            c.execute(sql, args)
    return n_files
