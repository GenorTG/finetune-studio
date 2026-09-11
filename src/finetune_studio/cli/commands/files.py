"""`fts files trash` — manual trash cleanup + list for the per-project file library.

Examples:
  fts files trash --list                       # list all trashed files across all projects
  fts files trash --list --project-id b08426e3 # list trash for one project
  fts files trash --older-than-days 7          # purge trashed files older than 7 days
  fts files trash --older-than-days 7 --dry-run # show what would be purged
  fts files trash --all                        # purge everything in trash (no age filter)
"""
from __future__ import annotations

import json
import sys
import time

from finetune_studio import db


def cmd_files(args) -> None:
    # Handle both top-level `fts files trash` (subcommand group) and bare
    # `fts files` (print usage).
    sub = getattr(args, "files_command", None)
    if sub is None or sub == "trash":
        _cmd_files_trash(args)
    else:
        print(f"Unknown files subcommand: {sub}")
        print("Usage: fts files trash [--older-than-days N] [--project-id PID] [--list] [--dry-run]")
        sys.exit(1)


def _cmd_files_trash(args) -> None:
    """List or purge trash. Args:
      --older-than-days N  : purge files whose deleted_at is older than N days
                             (default = 7). Pass 0 to purge all.
      --project-id PID     : restrict to a single project (else all projects)
      --list               : just list trash contents, do not purge
      --dry-run            : show what would be purged, do not delete
      --all                : purge everything in trash regardless of age
    """
    from finetune_studio.data.fs import file_library as fl

    # Decide which projects to act on.
    pid_filter = getattr(args, "project_id", None)
    if pid_filter:
        projects = [db.get_project(pid_filter)]
        if not projects[0]:
            print(f"Project not found: {pid_filter}")
            sys.exit(1)
    else:
        projects = db.list_projects()
    if not projects:
        print("No projects found.")
        return

    just_list = getattr(args, "list", False)
    purge_all = getattr(args, "all", False)
    dry_run = getattr(args, "dry_run", False)
    older_than = getattr(args, "older_than_days", 7)

    if purge_all:
        # Override age to 0 so we always purge.
        older_than = 0

    if just_list:
        for p in projects:
            items = fl.list_trash(p["id"])
            if not items:
                continue
            print(f"\n=== {p['name']} ({p['id']}) — {len(items)} trashed ===")
            for it in items:
                print(f"  {it['original_name']:<40} {it['mime_type']:<32} age={it['age_days']}d  deleted_at={it['deleted_at']:.0f}")
        return

    # Purge mode
    now = time.time()
    total_purged = 0
    for p in projects:
        if purge_all:
            # For "all" mode, override with 0 days = purge everything
            cutoff = now + 1  # everything is older than now+1s
            items = fl.list_trash(p["id"])
        else:
            cutoff = now - older_than * 86400
            items = [it for it in fl.list_trash(p["id"]) if it["deleted_at"] < cutoff]
        if not items:
            continue
        print(f"\n=== {p['name']} ({p['id']}) — {len(items)} to purge (older than {older_than}d) ===")
        for it in items:
            print(f"  {it['original_name']:<40} age={it['age_days']}d")
        if dry_run:
            total_purged += len(items)
            continue
        result = fl.purge_trash(p["id"], older_than_days=older_than)
        print(f"  → purged {result['count']} files")
        total_purged += result["count"]

    if dry_run:
        print(f"\n[dry-run] Would purge {total_purged} files across {len(projects)} project(s).")
    else:
        print(f"\nPurged {total_purged} files across {len(projects)} project(s).")
