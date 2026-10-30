"""Check, apply and inspect self-healing application updates."""
from __future__ import annotations

import argparse
import json
import sys
import time
from typing import Any

from finetune_studio import db
from finetune_studio.webui.routes import updates


class UpdateCommandError(Exception):
    """A user-facing update CLI failure."""


def _compact(row: dict[str, Any], *, full: bool = False) -> dict[str, Any]:
    out = dict(row)
    log_text = str(out.pop("log_text", "") or "")
    out["log_tail"] = log_text if full else log_text[-4000:]
    out["log_length"] = len(log_text)
    options = out.pop("options", None)
    out.pop("options_json", None)
    out["options"] = options if isinstance(options, dict) else {}
    return out


def _emit(value: Any) -> None:
    print(json.dumps(value, indent=2, ensure_ascii=False))


def _timestamp(value: float | None) -> str:
    return time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(value)) if value else "—"


def _print_update(row: dict[str, Any], *, full: bool = False) -> None:
    out = _compact(row, full=full)
    print(f"{out['id']} · {out['mode']} · {out['status']} · started {_timestamp(out.get('started_at'))}")
    if out.get("error"):
        print(f"Error: {out['error']}")
    log_text = out.get("log_tail") or ""
    if log_text:
        print(log_text, end="" if log_text.endswith("\n") else "\n")


def _history(args: argparse.Namespace) -> None:
    rows = [_compact(row) for row in db.list_updates_recent(limit=args.limit)]
    if args.json:
        _emit({"updates": rows})
        return
    if not rows:
        print("No update attempts yet.")
        return
    print(f"{'ID':<28} {'MODE':<8} {'STATUS':<10} CREATED")
    for row in rows:
        print(f"{row['id']:<28} {row['mode']:<8} {row['status']:<10} {_timestamp(row.get('created_at'))}")


def _status(args: argparse.Namespace) -> None:
    row = db.get_update(args.update_id)
    if row is None:
        raise UpdateCommandError(f"unknown update id {args.update_id!r}")
    out = _compact(row, full=args.full)
    if args.json:
        _emit(out)
    else:
        _print_update(row, full=args.full)


def _confirm(args: argparse.Namespace, mode: str) -> bool:
    if mode == "check" or args.yes:
        return True
    label = "Repair recreates the Python environment from scratch" if mode == "repair" else (
        "Apply installs code and packages, then restarts the service"
    )
    if not sys.stdin.isatty():
        raise UpdateCommandError(f"{label}; non-interactive use requires --yes")
    answer = input(f"{label}. Continue? [y/N] ")
    return answer.strip().casefold() in {"y", "yes"}


def _run(args: argparse.Namespace) -> None:
    mode = {"check": "check", "apply": "update", "repair": "repair"}[args.update_action]
    if not _confirm(args, mode):
        print("Cancelled.")
        return

    options = {
        "no_pull": args.no_pull,
        "no_llama": args.no_llama,
        "no_restart": args.no_restart,
    }
    row = db.create_update(mode=mode, options=options, triggered_by="cli")

    if not args.json:
        print(f"Update {row['id']} · {mode} · running")

    def on_log(line: str) -> None:
        if not args.json:
            print(line, end="" if line.endswith("\n") else "\n", flush=True)

    updates.run_update_worker(row["id"], mode, options, on_log=on_log)
    finished = db.get_update(row["id"])
    if finished is None:
        raise UpdateCommandError(f"update record {row['id']} disappeared while running")
    result = _compact(finished)
    if args.json:
        _emit(result)
    else:
        print(f"Update {row['id']} · {finished['status']}")
        if finished.get("error"):
            print(f"Error: {finished['error']}", file=sys.stderr)
    if finished["status"] != "done":
        raise SystemExit(1)


def cmd_update(args: argparse.Namespace) -> None:
    """Run the updater or inspect its durable history."""
    db.init_db()
    try:
        if args.update_action == "history":
            _history(args)
        elif args.update_action == "status":
            _status(args)
        else:
            _run(args)
    except UpdateCommandError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        raise SystemExit(1) from exc
