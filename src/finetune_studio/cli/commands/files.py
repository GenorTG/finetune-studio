"""Project-scoped CLI commands for the shared file library."""
from __future__ import annotations

import argparse
import json
import sys
import time
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
    """Handle `fts files {list,show,parse,trash}`."""
    db.init_db()
    sub = getattr(args, "files_command", None)
    try:
        if sub is None:
            print("Usage: fts files {list,show,parse,trash} ...")
            return
        if sub == "list":
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
