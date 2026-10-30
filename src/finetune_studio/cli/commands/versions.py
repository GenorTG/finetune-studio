"""Project snapshot history commands for the local ``fts`` CLI."""
from __future__ import annotations

import argparse
import json
import sys
from typing import Any

from finetune_studio import db
from finetune_studio.project_versioning import ProjectVersionError, save_project_version


class VersionCommandError(Exception):
    """A user-facing version-history CLI failure."""


def _resolve_project(ref: str) -> dict[str, Any]:
    project = db.get_project(ref)
    if project:
        return project
    projects = db.list_projects()
    exact = [row for row in projects if str(row.get("name") or "") == ref]
    matches = exact or [row for row in projects if str(row.get("name") or "").casefold() == ref.casefold()]
    if len(matches) == 1:
        return matches[0]
    if len(matches) > 1:
        ids = ", ".join(str(row["id"]) for row in matches)
        raise VersionCommandError(f"project name {ref!r} is ambiguous; use one of these ids: {ids}")
    raise VersionCommandError(f"project not found: {ref!r}")


def _owned_version(pid: str, version_id: str) -> dict[str, Any]:
    version = db.get_version(version_id)
    if not version or str(version.get("project_id")) != pid:
        raise VersionCommandError(f"version not found in this project: {version_id!r}")
    return version


def _decode(version: dict[str, Any]) -> dict[str, Any]:
    result = dict(version)
    try:
        result["manifest"] = json.loads(result.get("manifest_json") or "{}")
    except (TypeError, json.JSONDecodeError):
        result["manifest"] = {}
    result.pop("manifest_json", None)
    return result


def _list(args: argparse.Namespace) -> None:
    project = _resolve_project(args.project)
    versions = [_decode(row) for row in db.list_versions(str(project["id"]))]
    result = {"project": {"id": project["id"], "name": project["name"]}, "versions": versions}
    if args.json:
        print(json.dumps(result, indent=2))
        return
    if not versions:
        print(f"No pinned versions in {project['name']!r}.")
        return
    print(f"{'VERSION':<12} {'LABEL':<28} {'PARENT':<24} CREATED")
    print("-" * 92)
    for version in versions:
        created = float(version.get("created_at") or 0)
        parent = version.get("parent_version_id") or "—"
        print(f"v{version['version_number']:<11} {version.get('label') or '—'!s:<28} "
              f"{parent:<24} {created:.0f}")


def _pin(args: argparse.Namespace) -> None:
    project = _resolve_project(args.project)
    pid = str(project["id"])
    parent_id = args.parent
    if args.root:
        parent_id = None
    elif parent_id is None:
        latest = db.latest_version(pid)
        parent_id = str(latest["id"]) if latest else None
    else:
        _owned_version(pid, str(parent_id))

    try:
        version = save_project_version(pid, {"label": args.label, "notes": args.notes,
                                              "parent_version_id": parent_id})
    except ProjectVersionError as exc:
        raise VersionCommandError(exc.detail) from exc
    result = {"ok": True, "project": {"id": project["id"], "name": project["name"]},
              "version": _decode(version)}
    if args.json:
        print(json.dumps(result, indent=2))
    else:
        parent = f" building on {parent_id}" if parent_id else " as a new root"
        print(f"Pinned v{version['version_number']} {version.get('label') or ''}{parent}.")


def _show(args: argparse.Namespace) -> None:
    project = _resolve_project(args.project)
    version = _decode(_owned_version(str(project["id"]), args.version_id))
    result = {"project": {"id": project["id"], "name": project["name"]}, "version": version}
    if args.json:
        print(json.dumps(result, indent=2))
        return
    print(f"Version : v{version['version_number']} {version.get('label') or ''} [{version['id']}]")
    print(f"Created : {version.get('created_at')}")
    print(f"Parent  : {version.get('parent_version_id') or '—'}")
    if version.get("notes"):
        print(f"Notes   : {version['notes']}")
    print("Manifest:")
    print(json.dumps(version["manifest"], indent=2, ensure_ascii=False))


def _lineage(args: argparse.Namespace) -> None:
    project = _resolve_project(args.project)
    pid = str(project["id"])
    _owned_version(pid, args.version_id)
    chain = [_decode(row) for row in db.version_lineage(pid, args.version_id)]
    result = {"project": {"id": project["id"], "name": project["name"]}, "lineage": chain}
    if args.json:
        print(json.dumps(result, indent=2))
        return
    print(" → ".join(f"v{row['version_number']} {row.get('label') or row['id']}" for row in chain))


def _delete(args: argparse.Namespace) -> None:
    project = _resolve_project(args.project)
    pid = str(project["id"])
    version = _owned_version(pid, args.version_id)
    if not args.yes:
        prompt = (f"Delete version v{version['version_number']} {version.get('label') or ''}? "
                  "Only the saved record goes; datasets, runs, models and Brain indexes stay. [y/N] ")
        if not sys.stdin.isatty():
            raise VersionCommandError("deleting a version without a terminal requires --yes")
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

    db.delete_version(str(version["id"]))
    result = {"ok": True, "deleted": {"id": version["id"], "version_number": version["version_number"],
                                          "label": version.get("label")}}
    if args.json:
        print(json.dumps(result, indent=2))
    else:
        print(f"Deleted v{version['version_number']} record; referenced datasets and model artifacts remain.")


def cmd_versions(args: argparse.Namespace) -> None:
    """Handle ``fts versions {list,pin,show,lineage,delete}``."""
    db.init_db()
    actions = {"list": _list, "pin": _pin, "show": _show, "lineage": _lineage, "delete": _delete}
    try:
        actions[args.versions_action](args)
    except VersionCommandError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        raise SystemExit(1) from exc
