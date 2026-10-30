"""Project-scoped CLI commands for the shared file library."""
from __future__ import annotations

import argparse
import hashlib
import json
import mimetypes
import sys
import tempfile
import time
from pathlib import Path
from typing import Any

from finetune_studio import db
from finetune_studio.data.fs import file_library as fl
from finetune_studio.knowledge import sources
from finetune_studio.knowledge.errors import SourceError


class FileCommandError(Exception):
    """A user-facing Files CLI failure."""


def _resolve_project(ref: str) -> dict[str, Any]:
    projects = db.list_projects()
    by_id = next((p for p in projects if str(p.get("id") or "") == ref), None)
    if by_id:
        return by_id

    exact = [p for p in projects if str(p.get("name") or "") == ref]
    matches = exact or [p for p in projects if str(p.get("name") or "").casefold() == ref.casefold()]
    if len(matches) == 1:
        return matches[0]
    if len(matches) > 1:
        ids = ", ".join(str(p["id"]) for p in matches)
        raise FileCommandError(f"project name {ref!r} is ambiguous; use one of these ids: {ids}")
    raise FileCommandError(f"project not found: {ref!r}")


def _resolve_file(pid: str, ref: str, *, include_deleted: bool = False) -> dict[str, Any]:
    files = fl.list_files(pid, include_deleted=include_deleted)
    exact_id = next((item for item in files if item["id"] == ref), None)
    if exact_id:
        return exact_id

    matches = [
        item for item in files
        if str(item.get("original_name") or "").casefold() == ref.casefold()
    ]
    if len(matches) == 1:
        return matches[0]
    if len(matches) > 1:
        ids = ", ".join(str(item["id"]) for item in matches)
        raise FileCommandError(f"file name {ref!r} is ambiguous; use one of these ids: {ids}")
    raise FileCommandError(f"file not found in project: {ref!r}")


def _file_fields(file: dict[str, Any]) -> dict[str, Any]:
    keys = (
        "id", "original_name", "mime_type", "current_version", "size_bytes",
        "uploaded_at", "uploaded_by", "last_trained_at", "deleted_at", "tags",
        "notes", "first_folder", "folder_id",
    )
    return {key: file.get(key) for key in keys if key in file}


def _source_fields(pid: str, file_id: str) -> dict[str, Any] | None:
    source = sources.source_for_file(pid, file_id)
    if source is None:
        return None
    return {
        "id": source.id,
        "status": source.status,
        "parser": source.parser,
        "char_count": source.char_count,
        "chunk_count": source.chunk_count,
        "error": source.error,
        "edited_at": source.edited_at,
    }


def _list_files(args: argparse.Namespace) -> None:
    project = _resolve_project(args.project)
    files = fl.list_files(
        str(project["id"]),
        include_deleted=args.include_deleted,
        mime_prefix=args.mime,
        search=args.search,
    )
    result = []
    for file in files:
        row = _file_fields(file)
        row["parse"] = _source_fields(str(project["id"]), str(file["id"]))
        result.append(row)

    if args.json:
        print(json.dumps({"project": {"id": project["id"], "name": project["name"]}, "files": result}, indent=2))
        return
    if not result:
        print(f"No files found in {project['name']!r}.")
        return
    print(f"{'Name':<40} {'ID':<24} {'Size':>10}  {'Parse'}")
    print("-" * 96)
    for row in result:
        state = (row["parse"] or {}).get("status", "not staged")
        print(f"{row['original_name']:<40} {row['id']:<24} {row['size_bytes']:>10}  {state}")


def _show_file(args: argparse.Namespace) -> None:
    project = _resolve_project(args.project)
    file = _resolve_file(str(project["id"]), args.file, include_deleted=args.include_deleted)
    result = _file_fields(file)
    result["parse"] = _source_fields(str(project["id"]), str(file["id"]))
    if args.json:
        print(json.dumps(result, indent=2))
        return
    print(f"File       : {file['original_name']}")
    print(f"ID         : {file['id']}")
    print(f"Type       : {file['mime_type']}")
    print(f"Size       : {file['size_bytes']} bytes")
    print(f"Version    : {file['current_version']}")
    print(f"Parse      : {(result['parse'] or {}).get('status', 'not staged')}")


def _parse_file(args: argparse.Namespace) -> None:
    project = _resolve_project(args.project)
    file = _resolve_file(str(project["id"]), args.file)
    try:
        source = sources.stage(str(project["id"]), str(file["id"]))
        source = sources.parse(str(project["id"]), source.id, force=args.reparse)
    except SourceError as exc:
        raise FileCommandError(str(exc)) from exc

    result = {
        "file_id": file["id"],
        "name": file["original_name"],
        "source": {
            "id": source.id,
            "status": source.status,
            "parser": source.parser,
            "char_count": source.char_count,
            "chunk_count": source.chunk_count,
            "error": source.error,
        },
    }
    if args.json:
        print(json.dumps(result, indent=2))
    elif source.status == sources.ERROR:
        print(f"Could not parse {source.filename}: {source.error}", file=sys.stderr)
    else:
        action = "Reparsed" if args.reparse else "Parsed"
        print(f"{action} {source.filename!r}: {source.status} · {source.parser} · "
              f"{source.char_count} characters · {source.chunk_count} passages")
    if source.status == sources.ERROR:
        raise SystemExit(1)


def _stage_path(pid: str, path: Path) -> tuple[Path, int, str]:
    """Copy a local file into project staging while hashing it in bounded memory."""
    staging_dir = fl.project_files_root(pid) / ".uploads"
    staging_dir.mkdir(parents=True, exist_ok=True)
    digest = hashlib.sha256()
    size = 0
    with path.open("rb") as source_file, tempfile.NamedTemporaryFile(
        prefix="upload-", suffix=".part", dir=staging_dir, delete=False
    ) as staged:
        staged_path = Path(staged.name)
        try:
            while chunk := source_file.read(8 * 1024 * 1024):
                staged.write(chunk)
                digest.update(chunk)
                size += len(chunk)
        except Exception:
            staged_path.unlink(missing_ok=True)
            raise
    return staged_path, size, digest.hexdigest()


def _parse_uploaded_source(pid: str, file_id: str, filename: str) -> dict[str, Any] | None:
    """Parse supported uploads through the same source pipeline as the WebUI."""
    if not sources.is_parseable(filename):
        return None
    try:
        source = sources.stage(pid, file_id)
        source = sources.parse(pid, source.id)
    except SourceError as exc:
        return {"status": sources.ERROR, "error": str(exc)}
    return {
        "id": source.id,
        "status": source.status,
        "parser": source.parser,
        "char_count": source.char_count,
        "chunk_count": source.chunk_count,
        "error": source.error,
    }


def _upload_files(args: argparse.Namespace) -> None:
    project = _resolve_project(args.project)
    pid = str(project["id"])
    fl.ensure_dirs(pid)
    report: list[dict[str, Any]] = []
    counts = {"uploaded": 0, "duplicates_skipped": 0, "errors": 0}

    for raw_path in args.paths:
        path = Path(raw_path).expanduser()
        name = path.name or "unnamed"
        staged_path: Path | None = None
        try:
            if not path.is_file():
                raise FileNotFoundError(f"not a regular file: {path}")
            staged_path, size_bytes, raw_hash = _stage_path(pid, path)
            if not size_bytes:
                raise ValueError("empty upload")

            existing = fl.find_existing_hash(pid, raw_hash)
            if existing:
                staged_path.unlink(missing_ok=True)
                staged_path = None
                source = _parse_uploaded_source(pid, str(existing["id"]), str(existing["original_name"]))
                report.append({
                    "filename": name,
                    "status": "duplicate",
                    "duplicate_of": existing["original_name"],
                    "file_id": existing["id"],
                    "source": source,
                })
                counts["duplicates_skipped"] += 1
                continue

            mime_hint = mimetypes.guess_type(name)[0]
            meta = fl.write_staged_upload(
                pid=pid,
                staged_path=staged_path,
                original_name=name,
                raw_hash=raw_hash,
                size_bytes=size_bytes,
                mime_hint=mime_hint,
            )
            staged_path = None  # adopted by the library
            source = _parse_uploaded_source(pid, meta.file_id, name)
            report.append({
                "filename": name,
                "status": "uploaded",
                "file_id": meta.file_id,
                "mime_type": meta.mime_type,
                "size_bytes": meta.size_bytes,
                "raw_hash": meta.raw_hash,
                "auto_kind": meta.auto_kind,
                "source": source,
                "parse_status": source["status"] if source else "not staged",
            })
            counts["uploaded"] += 1
        except Exception as exc:  # noqa: BLE001 - isolate bulk uploads like the WebUI route
            if staged_path is not None:
                staged_path.unlink(missing_ok=True)
            report.append({"filename": name, "status": "error", "error": str(exc)})
            counts["errors"] += 1

    result = {"ok": counts["errors"] == 0, "project": {"id": pid, "name": project["name"]},
              "counts": counts, "report": report}
    if args.json:
        print(json.dumps(result, indent=2))
    else:
        for item in report:
            if item["status"] == "error":
                print(f"Error uploading {item['filename']!r}: {item['error']}", file=sys.stderr)
            elif item["status"] == "duplicate":
                print(f"Skipped duplicate {item['filename']!r}; already stored as {item['duplicate_of']!r}")
            else:
                source = item.get("source")
                parse_note = f" · parse {source['status']}" if source else " · no parser for this file type"
                print(f"Uploaded {item['filename']!r} [{item['file_id']}] · {item['size_bytes']} bytes{parse_note}")
        print(f"Uploaded {counts['uploaded']}; skipped duplicates {counts['duplicates_skipped']}; errors {counts['errors']}.")
    if counts["errors"]:
        raise SystemExit(1)


def _cmd_files_trash(args: argparse.Namespace) -> None:
    """List or purge trash; purge is the existing explicit maintenance command."""
    pid_filter = getattr(args, "project_id", None)
    if pid_filter:
        projects = [db.get_project(pid_filter)]
        if not projects[0]:
            raise FileCommandError(f"project not found: {pid_filter!r}")
    else:
        projects = db.list_projects()
    if not projects:
        print("No projects found.")
        return

    just_list = getattr(args, "list", False)
    purge_all = getattr(args, "all", False)
    dry_run = getattr(args, "dry_run", False)
    older_than = getattr(args, "older_than_days", 7)
    if older_than < 0:
        raise FileCommandError("--older-than-days must be zero or greater")
    if purge_all:
        older_than = 0

    if just_list:
        for project in projects:
            items = fl.list_trash(project["id"])
            if not items:
                continue
            print(f"\n=== {project['name']} ({project['id']}) — {len(items)} trashed ===")
            for item in items:
                print(f"  {item['original_name']:<40} {item['mime_type']:<32} "
                      f"age={item['age_days']}d  deleted_at={item['deleted_at']:.0f}")
        return

    now = time.time()
    total = 0
    for project in projects:
        if purge_all:
            items = fl.list_trash(project["id"])
        else:
            cutoff = now - older_than * 86400
            items = [item for item in fl.list_trash(project["id"]) if item["deleted_at"] < cutoff]
        if not items:
            continue
        print(f"\n=== {project['name']} ({project['id']}) — {len(items)} to purge "
              f"(older than {older_than}d) ===")
        for item in items:
            print(f"  {item['original_name']:<40} age={item['age_days']}d")
        if dry_run:
            total += len(items)
            continue
        result = fl.purge_trash(project["id"], older_than_days=older_than)
        print(f"  → purged {result['count']} files")
        total += result["count"]

    if dry_run:
        print(f"\n[dry-run] Would purge {total} files across {len(projects)} project(s).")
    else:
        print(f"\nPurged {total} files across {len(projects)} project(s).")


def cmd_files(args: argparse.Namespace) -> None:
    """Handle `fts files {upload,list,show,parse,trash}`."""
    db.init_db()
    sub = getattr(args, "files_command", None)
    try:
        if sub is None:
            print("Usage: fts files {upload,list,show,parse,trash} ...")
            return
        if sub == "upload":
            _upload_files(args)
        elif sub == "list":
            _list_files(args)
        elif sub == "show":
            _show_file(args)
        elif sub == "parse":
            _parse_file(args)
        elif sub == "trash":
            _cmd_files_trash(args)
        else:
            raise FileCommandError(f"unknown files command: {sub}")
    except FileCommandError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        raise SystemExit(1) from exc
