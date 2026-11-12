"""Q&A review queue and verdicts for the local ``fts`` CLI."""
from __future__ import annotations

import argparse
import json
import sys
import time
from typing import Any

from finetune_studio import db
from finetune_studio.data.fs import qa as qa_fs
from finetune_studio.data.fs.qa_review import status_counts


class QACommandError(Exception):
    """A user-facing Q&A review CLI failure."""


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
        raise QACommandError(f"project name {ref!r} is ambiguous; use one of these ids: {ids}")
    raise QACommandError(f"project not found: {ref!r}")


def _resolve_source(pid: str, ref: str | None) -> str | None:
    if not ref:
        return None
    from finetune_studio.knowledge.sources import list_sources

    sources = list_sources(pid)
    matches = [source for source in sources if source.id == ref]
    if not matches:
        matches = [source for source in sources if source.filename == ref]
    if not matches:
        matches = [source for source in sources if source.filename.casefold() == ref.casefold()]
    if len(matches) > 1:
        ids = ", ".join(source.id for source in matches)
        raise QACommandError(f"file name {ref!r} is ambiguous; use one of these source ids: {ids}")
    if not matches:
        raise QACommandError(f"parsed file not found in project: {ref!r}")
    return matches[0].id


def _list(args: argparse.Namespace) -> None:
    project = _resolve_project(args.project)
    from finetune_studio.data.fs.qa_review import review_queue

    source_id = _resolve_source(str(project["id"]), args.source) if args.source else None
    result = {"project": {"id": project["id"], "name": project["name"]},
              **review_queue(str(project["id"]), status=args.status, source_id=source_id, text=args.search,
                             offset=args.offset, limit=args.limit)}
    if args.json:
        print(json.dumps(result, indent=2))
        return
    counts = result["counts"]
    print(f"{project['name']}: {result['total']} pair(s) · pending {counts['pending']}, "
          f"approved {counts['approved']}, rejected {counts['rejected']}")
    if not result["items"]:
        print("No matching Q&A pairs.")
        return
    for item in result["items"]:
        print(f"[{item['id']}] {item['status']} · {item.get('source_filename') or item.get('source_id')} "
              f"· chunk {item.get('chunk_idx', '?')}")
        print(f"  Q: {item.get('question', '')}")
        print(f"  A: {item.get('answer', '')}")


def _last_coverage_warnings(pid: str, selected_ids: set[str]) -> list[dict[str, str]]:
    """Find items that would have no approved or pending pair after this rejection batch."""
    from finetune_studio.data.mining.coverage import item_states

    pairs = qa_fs.list_qa_pairs(pid)
    selected = {str(pair.get("id")) for pair in pairs if str(pair.get("id")) in selected_ids}
    warnings = []
    sources = {str(pair.get("source_id") or "") for pair in pairs if str(pair.get("id")) in selected}
    for source_id in sources:
        file_pairs = [pair for pair in pairs if str(pair.get("source_id") or "") == source_id]
        items = db.mining.list_items(pid, source_id)
        before = {state.item_id: state for state in item_states(items, file_pairs)}
        after_pairs = [
            {**pair, "status": "rejected"} if str(pair.get("id")) in selected else pair
            for pair in file_pairs
        ]
        after = {state.item_id: state for state in item_states(items, after_pairs)}
        for item_id, state in before.items():
            next_state = after.get(item_id)
            if state.state in {"approved", "pending"} and next_state and next_state.state == "open":
                warnings.append({"source_id": source_id, "item_id": item_id})
    return warnings


def _change_status(args: argparse.Namespace, status: str) -> None:
    project = _resolve_project(args.project)
    pairs = qa_fs.list_qa_pairs(str(project["id"]))
    by_id = {str(pair.get("id")): pair for pair in pairs}
    ids = list(dict.fromkeys(args.ids))
    missing = [qa_id for qa_id in ids if qa_id not in by_id]
    if missing:
        raise QACommandError(f"Q&A pair(s) not found in this project: {', '.join(missing)}")

    warnings = _last_coverage_warnings(str(project["id"]), set(ids)) if status == "rejected" else []
    if warnings and not getattr(args, "yes", False):
        detail = "; ".join(f"{row['source_id']} data point {row['item_id']} would be left uncovered" for row in warnings)
        prompt = f"Rejecting these pairs leaves {detail}. Continue? [y/N] "
        if not sys.stdin.isatty():
            raise QACommandError(f"{detail}; rerun with --yes to confirm")
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

    now = time.time()
    updated = [qa_fs.update_qa_pair(str(project["id"]), qa_id, status=status, reviewed_at=now,
                                    **({"note": args.note} if args.note is not None else {})) for qa_id in ids]
    updated_ids = [str(pair["id"]) for pair in updated if pair is not None]
    result = {"ok": True, "project": {"id": project["id"], "name": project["name"]},
              "status": status, "updated": len(updated_ids), "updated_ids": updated_ids,
              "counts": status_counts(qa_fs.list_qa_pairs(str(project["id"]))),
              "coverage_warnings": warnings}
    if args.json:
        print(json.dumps(result, indent=2))
    else:
        print(f"Marked {result['updated']} pair(s) {status}. Counts: "
              f"pending {result['counts']['pending']}, approved {result['counts']['approved']}, "
              f"rejected {result['counts']['rejected']}.")


def cmd_qa(args: argparse.Namespace) -> None:
    """Handle ``fts qa {list,approve,reject}``; use ``fts dataset build`` to export."""
    db.init_db()
    try:
        if args.qa_action == "list":
            _list(args)
        elif args.qa_action == "approve":
            _change_status(args, "approved")
        elif args.qa_action == "reject":
            _change_status(args, "rejected")
        else:
            raise QACommandError(f"unknown QA command: {args.qa_action}")
    except QACommandError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        raise SystemExit(1) from exc
