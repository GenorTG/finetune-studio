"""`fts models` — list installed models and manage Hugging Face downloads."""
from __future__ import annotations

import argparse
import json
import sys


class ModelCommandError(Exception):
    """A user-facing model-library CLI failure."""


def cmd_models(args) -> None:
    if getattr(args, "models_action", None):
        actions = {
            "search": _search,
            "local": _local,
            "download": _download,
            "download-status": _download_status,
            "cancel-download": _cancel_download,
            "delete": _delete,
        }
        try:
            actions[args.models_action](args)
        except ModelCommandError as exc:
            print(f"Error: {exc}", file=sys.stderr)
            raise SystemExit(1) from exc
        return

    _list_installed(args)


def _list_installed(args: argparse.Namespace) -> None:
    from finetune_studio.config import settings
    from finetune_studio.models.registry import scan_models
    # Same directory set the WebUI scans (app.py lifespan / routes/models.py).
    dirs = list(settings.model_dirs)
    for d in [*settings.model_dirs_extra, *(args.dirs or [])]:
        if d not in dirs:
            dirs.append(d)
    models = scan_models(dirs)
    if args.json:
        print(json.dumps([{
            "name": m.name, "path": m.path, "format": m.format,
            "size_gb": m.size_gb, "architecture": m.architecture,
        } for m in models], indent=2))
        return
    if not models:
        print("No models found.")
        return
    print(f"{'Name':<50} {'Format':<12} {'Size':<10} {'Arch'}")
    print("-" * 90)
    for m in models:
        print(f"{m.name:<50} {m.format:<12} {m.size_gb:<10} {m.architecture}")


def _search(args: argparse.Namespace) -> None:
    from finetune_studio.webui.routes.hf_models import (
        HFUnavailableError,
        SearchRequest,
        _search_hf,
    )

    if not 1 <= args.limit <= 100:
        raise ModelCommandError("--limit must be between 1 and 100")
    try:
        rows = _search_hf(SearchRequest(query=args.query, task=args.task, library=args.library,
                                        sort=args.sort, limit=args.limit))
    except HFUnavailableError as exc:
        raise ModelCommandError(f"Hugging Face Hub is unreachable: {exc}") from exc
    if args.json:
        print(json.dumps({"query": args.query, "task": args.task, "results": rows}, indent=2))
        return
    if not rows:
        print("No Hugging Face models found.")
        return
    print(f"{'Repository':<54} {'Downloads':>12} {'Likes':>10}  Tags")
    print("-" * 100)
    for row in rows:
        print(f"{row['repo_id']:<54} {row['downloads']:>12} {row['likes']:>10}  "
              f"{', '.join(row['tags'][:3])}")


def _local(args: argparse.Namespace) -> None:
    from finetune_studio import db
    from finetune_studio.webui.routes.hf_models import _local_models

    db.init_db()
    rows = _local_models()
    if args.json:
        print(json.dumps({"models": rows}, indent=2))
        return
    if not rows:
        print("No downloaded Hugging Face models.")
        return
    print(f"{'Repository / cache key':<54} {'Size':>12} {'Files':>7}  Path")
    print("-" * 110)
    for row in rows:
        identity = str(row['hub_repo_id']) if row.get('repo_id_known') else f"{row['cache_key']} (id unknown)"
        print(f"{identity:<54} {row['size_bytes'] / (1024 ** 3):>10.2f} GB "
              f"{row['file_count']:>7}  {row['path']}")


def _download(args: argparse.Namespace) -> None:
    from fastapi import HTTPException

    from finetune_studio import db
    from finetune_studio.jobs import launch
    from finetune_studio.webui import download_jobs
    from finetune_studio.webui.routes.hf_models import _local_dir

    db.init_db()
    try:
        row = download_jobs.start(repo_id=args.repo_id, filename=args.file,
                                  revision=args.revision, dest=str(_local_dir(args.repo_id)))
    except (launch.JobRejected, ValueError, HTTPException) as exc:
        raise ModelCommandError(f"could not start download: {exc}") from exc
    result = {"ok": True, "job_id": row["id"], "repo_id": args.repo_id,
              "started_at": row.get("started_at")}
    if args.json:
        print(json.dumps(result, indent=2))
    else:
        print(f"Download queued for {args.repo_id}. Job id: {row['id']}\n"
              f"Check: fts models download-status {row['id']}")


def _download_status(args: argparse.Namespace) -> None:
    from finetune_studio import db
    from finetune_studio.webui import download_jobs

    db.init_db()
    progress = download_jobs.progress(args.job_id)
    if progress is None:
        raise ModelCommandError(f"download job not found: {args.job_id}")
    if args.json:
        print(json.dumps(progress, indent=2))
        return
    print(f"{progress.get('status', 'unknown').upper()}: {progress.get('repo_id', 'model')} — "
          f"{progress.get('bytes_done', 0)} / {progress.get('bytes_total') or '?'} bytes")
    if progress.get("error"):
        print(f"Error: {progress['error']}")
    if progress.get("path"):
        print(f"Path: {progress['path']}")


def _cancel_download(args: argparse.Namespace) -> None:
    from finetune_studio import db
    from finetune_studio.webui import download_jobs

    db.init_db()
    status = download_jobs.cancel(args.job_id)
    if status is None:
        raise ModelCommandError(f"download job not found: {args.job_id}")
    result = {"ok": True, "job_id": args.job_id, "status": status}
    print(json.dumps(result, indent=2) if args.json else f"Download {args.job_id}: {status}.")


def _delete(args: argparse.Namespace) -> None:
    if not args.yes:
        prompt = (f"Delete downloaded model {args.repo_id!r} and all its files? "
                  "This cannot be undone. [y/N] ")
        if not sys.stdin.isatty():
            raise ModelCommandError("deleting a downloaded model without a terminal requires --yes")
        if args.json:
            print(prompt, file=sys.stderr, end="")
            answer = input()
        else:
            answer = input(prompt)
        if answer.strip().casefold() not in {"y", "yes"}:
            print(json.dumps({"ok": False, "cancelled": True}, indent=2) if args.json else "Cancelled.")
            return

    from finetune_studio import db
    from finetune_studio.models.hf_cache import HFCacheError, delete_cached_model

    db.init_db()
    try:
        target = delete_cached_model(args.repo_id)
    except HFCacheError as exc:
        raise ModelCommandError(str(exc)) from exc
    result = {"ok": True, "deleted": args.repo_id}
    print(json.dumps(result, indent=2) if args.json else f"Deleted {args.repo_id} ({target}).")
