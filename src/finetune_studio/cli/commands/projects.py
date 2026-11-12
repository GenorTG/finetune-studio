"""CLI commands for local project records (list, show and create)."""
from __future__ import annotations

import argparse
import json
import sys
from typing import Any

from finetune_studio import db


class ProjectCommandError(Exception):
    """A user-facing project CLI failure."""


def _project_fields(project: dict[str, Any]) -> dict[str, Any]:
    """Return stable public project fields without internal query details."""
    keys = ("id", "name", "description", "base_model", "system_prompt", "created_at", "updated_at")
    return {key: project.get(key) for key in keys}


def _project_list_fields(project: dict[str, Any]) -> dict[str, Any]:
    """Keep list output compact; use `show` for the full prompt and description."""
    return {key: project.get(key) for key in ("id", "name", "base_model", "updated_at")}


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
        raise ProjectCommandError(f"project name {ref!r} is ambiguous; use one of these ids: {ids}")
    raise ProjectCommandError(f"project not found: {ref!r}")


def _list(args: argparse.Namespace) -> None:
    db.init_db()
    projects = [_project_list_fields(p) for p in db.list_projects()]
    if args.json:
        print(json.dumps(projects, indent=2))
        return
    if not projects:
        print("No projects found.")
        return
    print(f"{'Name':<36} {'ID':<10} {'Base model'}")
    print("-" * 90)
    for project in projects:
        print(f"{project['name']:<36} {project['id']:<10} {project['base_model'] or '—'}")


def _show(args: argparse.Namespace) -> None:
    db.init_db()
    project = _project_fields(_resolve_project(args.project))
    if args.json:
        print(json.dumps(project, indent=2))
        return
    print(f"Project      : {project['name']}")
    print(f"ID           : {project['id']}")
    print(f"Description  : {project['description'] or '—'}")
    print(f"Base model   : {project['base_model'] or '—'}")
    print(f"System prompt: {project['system_prompt'] or '—'}")


def _create(args: argparse.Namespace) -> None:
    db.init_db()
    name = (args.name or "").strip()
    if not name:
        raise ProjectCommandError("--name must not be empty")
    if not args.allow_duplicate and any(
        str(p.get("name") or "").strip().casefold() == name.casefold() for p in db.list_projects()
    ):
        raise ProjectCommandError(f'a project named "{name}" already exists; choose a different name')

    project = _project_fields(db.create_project(
        name=name,
        description=args.description,
        base_model=args.base_model,
        system_prompt=args.system_prompt,
    ))
    if args.json:
        print(json.dumps(project, indent=2))
    else:
        print(f"Created project {project['name']!r} [{project['id']}]")


def _invalidate_model_registry() -> None:
    """Tell the running WebUI process to rescan its in-memory model list."""
    from finetune_studio.models.registry_invalidation import invalidate_model_registry

    invalidate_model_registry()


def _delete(args: argparse.Namespace) -> None:
    db.init_db()
    project = _resolve_project(args.project)
    if not args.yes:
        if not sys.stdin.isatty():
            raise ProjectCommandError("deleting a project without a terminal requires --yes")
        prompt = f"Delete project {project['name']!r} [{project['id']}] and all its stored data? [y/N] "
        if args.json:
            print(prompt, file=sys.stderr, end="")
            answer = input()
        else:
            answer = input(prompt)
        if answer.strip().casefold() not in {"y", "yes"}:
            if args.json:
                print(json.dumps({"ok": False, "cancelled": True}, indent=2))
            else:
                print("Cancelled.")
            return

    from finetune_studio.project_lifecycle import delete_project

    if not delete_project(str(project["id"])):
        raise ProjectCommandError(f"project not found: {args.project!r}")
    try:
        _invalidate_model_registry()
    except Exception as exc:  # noqa: BLE001 — deletion succeeded; stale registry is recoverable by rescan
        print(f"Warning: project deleted, but model registry refresh failed: {exc}", file=sys.stderr)

    result = {"ok": True, "project": _project_fields(project)}
    if args.json:
        print(json.dumps(result, indent=2))
    else:
        print(f"Deleted project {project['name']!r} [{project['id']}] and its stored data.")


def cmd_projects(args: argparse.Namespace) -> None:
    """Handle `fts projects {list,show,create,delete}`."""
    try:
        if args.projects_action == "list":
            _list(args)
        elif args.projects_action == "show":
            _show(args)
        elif args.projects_action == "create":
            _create(args)
        elif args.projects_action == "delete":
            _delete(args)
        else:
            raise ProjectCommandError(f"unknown projects command: {args.projects_action}")
    except ProjectCommandError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        raise SystemExit(1) from exc
