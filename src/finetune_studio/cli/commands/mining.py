"""Persistent per-file Mining jobs for the local ``fts`` CLI."""
from __future__ import annotations

import argparse
import json
import sys
from typing import Any

from finetune_studio import db


class MiningCommandError(Exception):
    """A user-facing Mining CLI failure."""


def _resolve_project(ref: str) -> dict[str, Any]:
    project = db.get_project(ref)
    if project:
        return project
    projects = db.list_projects()
    exact = [p for p in projects if str(p.get("name") or "") == ref]
    matches = exact or [p for p in projects if str(p.get("name") or "").casefold() == ref.casefold()]
    if len(matches) == 1:
        return matches[0]
    if len(matches) > 1:
        ids = ", ".join(str(p["id"]) for p in matches)
        raise MiningCommandError(f"project name {ref!r} is ambiguous; use one of these ids: {ids}")
    raise MiningCommandError(f"project not found: {ref!r}")


def _resolve_sources(pid: str, refs: list[str]) -> list[str]:
    from finetune_studio.knowledge.sources import list_sources

    available = list_sources(pid)
    resolved = []
    for ref in refs:
        matches = [source for source in available if source.id == ref]
        if not matches:
            matches = [source for source in available if source.filename == ref]
        if not matches:
            matches = [source for source in available if source.filename.casefold() == ref.casefold()]
        if len(matches) > 1:
            ids = ", ".join(source.id for source in matches)
            raise MiningCommandError(f"file name {ref!r} is ambiguous; use one of these source ids: {ids}")
        if not matches:
            raise MiningCommandError(f"parsed file not found in project: {ref!r}")
        if matches[0].id not in resolved:
            resolved.append(matches[0].id)
    return resolved


def _start(args: argparse.Namespace) -> None:
    project = _resolve_project(args.project)
    source_ids = _resolve_sources(str(project["id"]), args.files)
    from finetune_studio.data.mining import MiningError, get_scheduler

    try:
        jobs = get_scheduler().submit(str(project["id"]), source_ids, args.kind)
    except MiningError as exc:
        raise MiningCommandError(str(exc)) from exc
    result = {"ok": True, "project": {"id": project["id"], "name": project["name"]}, "jobs": jobs}
    if args.json:
        print(json.dumps(result, indent=2))
    else:
        for job in jobs:
            print(f"Queued {job['kind']} for {job['filename']!r} [{job['id']}] · {job['status']}")


def _list(args: argparse.Namespace) -> None:
    project = _resolve_project(args.project)
    from finetune_studio.data.mining.scheduler import helper_info

    limit = max(1, min(int(args.limit), 500))
    jobs = db.mining.list_jobs(str(project["id"]), source_id=args.source_id, active=args.active, limit=limit)
    result = {"project": {"id": project["id"], "name": project["name"]},
              "jobs": jobs, "helper": helper_info()}
    if args.json:
        print(json.dumps(result, indent=2))
        return
    if not jobs:
        print(f"No {'active ' if args.active else ''}Mining jobs in {project['name']!r}.")
        return
    print(f"{'ID':<24} {'STATUS':<12} {'STAGE':<14} {'KIND':<6} FILE")
    print("-" * 100)
    for job in jobs:
        print(f"{job['id']:<24} {job['status']:<12} {job['stage']:<14} {job['kind']:<6} {job['filename']}")


def _get_owned_job(pid: str, job_id: str) -> dict[str, Any]:
    job = db.mining.get_job(job_id)
    if not job or job.get("project_id") != pid:
        raise MiningCommandError(f"Mining job not found in this project: {job_id!r}")
    return job


def _status(args: argparse.Namespace) -> None:
    project = _resolve_project(args.project)
    job = _get_owned_job(str(project["id"]), args.job_id)
    if args.json:
        print(json.dumps(job, indent=2))
    else:
        print(f"Job      : {job['id']}")
        print(f"File     : {job['filename']} [{job['source_id']}]")
        print(f"Status   : {job['status']} · {job['stage']}")
        print(f"Progress : {job.get('message') or 'No progress message yet.'}")
        if job.get("error"):
            print(f"Error    : {job['error']}")


def _cancel(args: argparse.Namespace) -> None:
    project = _resolve_project(args.project)
    _get_owned_job(str(project["id"]), args.job_id)
    from finetune_studio.data.mining import MiningError, get_scheduler

    try:
        job = get_scheduler().cancel(str(project["id"]), args.job_id)
    except MiningError as exc:
        raise MiningCommandError(str(exc)) from exc
    if args.json:
        print(json.dumps({"ok": True, "job": job}, indent=2))
    elif job["status"] == "cancelled":
        print(f"Cancelled queued Mining job {job['id']}.")
    else:
        print(f"Stop requested for {job['id']} · {job['status']} · {job.get('message') or ''}")


def cmd_mining(args: argparse.Namespace) -> None:
    """Handle ``fts mining {start,list,status,cancel}``."""
    db.init_db()
    action = args.mining_action
    try:
        if action == "start":
            _start(args)
        elif action == "list":
            _list(args)
        elif action == "status":
            _status(args)
        elif action == "cancel":
            _cancel(args)
        else:
            raise MiningCommandError(f"unknown mining command: {action}")
    except MiningCommandError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        raise SystemExit(1) from exc
