"""Start and inspect the same supervised trained-model exports as the WebUI."""
from __future__ import annotations

import argparse
import json
import sys
from typing import Any

from finetune_studio import db
from finetune_studio.webui import export_service


class ExportCommandError(Exception):
    """A user-facing export CLI failure."""


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
        raise ExportCommandError(f"project name {ref!r} is ambiguous; use one of these ids: {ids}")
    raise ExportCommandError(f"project not found: {ref!r}")


def _emit(payload: dict[str, Any], as_json: bool) -> None:
    if as_json:
        print(json.dumps(payload, indent=2, ensure_ascii=False))
        return
    if payload.get("ok") is False:
        raise ExportCommandError(str(payload.get("detail") or "export request failed"))
    if "export_id" in payload:
        print(f"Export {payload['status']}: {payload['export_id']} · {payload.get('format')} · {payload.get('quant')}")
        if payload.get("output_path"):
            print(f"Output: {payload['output_path']}")


def _start(args: argparse.Namespace) -> None:
    project = _resolve_project(args.project)
    body: dict[str, Any] = {
        "format": args.format,
        "auto_merge": not args.no_auto_merge,
        "force": args.force,
        "imatrix": args.imatrix,
    }
    if args.base_model:
        body["base_model"] = args.base_model
    if args.quants:
        body["quants"] = args.quants
    elif args.quant:
        if args.imatrix:
            body["quants"] = [args.quant]
        else:
            body["quant"] = args.quant
    elif args.format.casefold() == "gguf":
        body["quants"] = [export_service.DEFAULT_QUANT]

    result = export_service.submit_export(str(project["id"]), args.run_id, body)
    if result.status_code >= 400:
        if args.json:
            _emit(result.body, True)
            raise SystemExit(1)
        raise ExportCommandError(str(result.body.get("detail") or "export request failed"))
    _emit({**result.body, "project": {"id": project["id"], "name": project["name"]}}, args.json)


def _list(args: argparse.Namespace) -> None:
    project = _resolve_project(args.project)
    limit = int(args.limit)
    if not 1 <= limit <= 1000:
        raise ExportCommandError("limit must be between 1 and 1000")
    rows = export_service.list_exports(str(project["id"]), limit=limit)
    payload = {"project": {"id": project["id"], "name": project["name"]}, "exports": rows}
    if args.json:
        _emit(payload, True)
        return
    if not rows:
        print(f"No exports in {project['name']!r}.")
        return
    print(f"{'ID':<25} {'STATUS':<12} {'FORMAT':<13} {'QUANT':<10} OUTPUT")
    print("-" * 110)
    for row in rows:
        print(f"{row['id']:<25} {row.get('status') or '?':<12} {row.get('format') or '?':<13} "
              f"{row.get('quant') or '—':<10} {row.get('output_path') or row.get('error') or ''}")


def _status(args: argparse.Namespace) -> None:
    project = _resolve_project(args.project)
    result = export_service.get_export(str(project["id"]), args.export_id)
    if result.status_code != 200:
        raise ExportCommandError(f"export not found in this project: {args.export_id!r}")
    if args.json:
        _emit(result.body, True)
        return
    row = result.body
    print(f"Export   : {row['id']}")
    print(f"Status   : {row.get('status') or '?'} · {row.get('phase') or 'waiting'}")
    print(f"Progress : {row.get('phase_detail') or 'No progress message yet.'}")
    if row.get("output_path"):
        print(f"Output   : {row['output_path']}")
    if row.get("error"):
        print(f"Error    : {row['error']}")


def _cancel(args: argparse.Namespace) -> None:
    project = _resolve_project(args.project)
    result = export_service.cancel_export(str(project["id"]), args.export_id)
    if result.status_code != 200:
        raise ExportCommandError(str(result.body.get("detail") or "export could not be cancelled"))
    _emit(result.body, args.json)
    if not args.json:
        print(f"Stop requested for export {args.export_id}.")


def cmd_export(args: argparse.Namespace) -> None:
    """Handle ``fts export {start,list,status,cancel}``."""
    db.init_db()
    try:
        if args.export_action == "start":
            _start(args)
        elif args.export_action == "list":
            _list(args)
        elif args.export_action == "status":
            _status(args)
        elif args.export_action == "cancel":
            _cancel(args)
        else:
            raise ExportCommandError(f"unknown export command: {args.export_action}")
    except ExportCommandError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        raise SystemExit(1) from exc
