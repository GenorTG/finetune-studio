"""Build, inspect, and evaluate the same template-based use cases as the Training page."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

from finetune_studio import db
from finetune_studio.usecases import evaluate, rows, store, teacher
from finetune_studio.usecases.schema import SchemaError
from finetune_studio.usecases.templates import TEMPLATES


class UsecaseCommandError(Exception):
    """A user-facing use-case CLI failure."""


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
        raise UsecaseCommandError(f"project name {ref!r} is ambiguous; use one of these ids: {ids}")
    raise UsecaseCommandError(f"project not found: {ref!r}")


def _owned(pid: str, uid: str) -> dict[str, Any]:
    try:
        return store.get(pid, uid)
    except LookupError as exc:
        raise UsecaseCommandError(f"use case not found in this project: {uid!r}") from exc


def _print(payload: Any, as_json: bool) -> None:
    if as_json:
        print(json.dumps(payload, indent=2, ensure_ascii=False))


def _templates(args: argparse.Namespace) -> None:
    templates = [template.as_dict() for template in TEMPLATES.values()]
    if args.json:
        _print({"templates": templates}, True)
        return
    for template in templates:
        needs = template["needs"]
        print(f"{template['id']}: {template['title']} · {template['scoring']} · "
              f"minimum {template['minimum_rows']} rows ({needs['minimum']} train + held-out)")
        print(f"  Example: {json.dumps(template['example_row'], ensure_ascii=False)}")


def _build(args: argparse.Namespace) -> None:
    project = _resolve_project(args.project)
    path = Path(args.file).expanduser()
    examples = rows.parse_upload(path.read_bytes())
    schema = json.loads(Path(args.schema).expanduser().read_text(encoding="utf-8")) if args.schema else None
    if schema is not None and not isinstance(schema, dict):
        raise UsecaseCommandError("schema file must contain a JSON object")

    teacher_report = None
    teacher_name = args.teacher.strip()
    if args.teacher_provider and args.template != "distillation":
        raise UsecaseCommandError("--teacher-provider is only valid for the distillation template")
    if args.teacher_provider:
        try:
            chat, teacher_name = teacher.open_teacher(args.teacher_provider.strip())
        except teacher.TeacherUnavailable as exc:
            raise UsecaseCommandError(str(exc)) from exc
        examples, report = teacher.fill_answers(examples, chat, args.instruction.strip())
        report.model = teacher_name
        teacher_report = report.as_dict()
        if report.asked and not report.answered:
            raise UsecaseCommandError(f"the teacher did not answer any prompts: {report.errors[:3]}")

    options = rows.BuildOptions(
        instruction=args.instruction,
        schema=schema,
        labels=[label.strip() for label in (args.labels or "").split(",") if label.strip()],
        style_guide=args.style_guide,
        teacher=teacher_name,
        seed=args.seed,
    )
    try:
        prepared = rows.prepare(args.template, examples, options)
    except (SchemaError, rows.RowsError) as exc:
        details = getattr(exc, "errors", [])
        suffix = "\n" + "\n".join(details) if details else ""
        raise UsecaseCommandError(f"{exc}{suffix}") from exc
    manifest = store.create(str(project["id"]), args.name, prepared, teacher_report=teacher_report)
    result = {
        "ok": True,
        "project": {"id": project["id"], "name": project["name"]},
        "usecase": manifest,
        "train_url": f"/projects/{project['id']}/training?usecase={manifest['id']}",
    }
    if args.json:
        _print(result, True)
        return
    print(f"Built {manifest['name']!r} [{manifest['id']}] · {manifest['train_rows']} training rows · "
          f"{manifest['heldout']} held-out questions")
    print(f"Training: {result['train_url']}")
    for warning in manifest.get("warnings") or []:
        print(f"Warning: {warning}")
    if teacher_report and teacher_report.get("failed"):
        print(f"Teacher answered {teacher_report['answered']}/{teacher_report['asked']}; "
              f"{teacher_report['failed']} prompt(s) were dropped.")


def _list(args: argparse.Namespace) -> None:
    project = _resolve_project(args.project)
    manifests = store.list_all(str(project["id"]))
    if args.json:
        _print({"project": {"id": project["id"], "name": project["name"]}, "usecases": manifests}, True)
        return
    if not manifests:
        print(f"No use cases in {project['name']!r}.")
        return
    print(f"{'ID':<12} {'TEMPLATE':<16} {'NAME':<32} {'ROWS':>8} EVALUATIONS")
    print("-" * 92)
    for manifest in manifests:
        print(f"{manifest['id']:<12} {manifest['template']:<16} {manifest['name'][:32]:<32} "
              f"{manifest.get('train_rows', 0):>8} {len(manifest.get('evaluations') or [])}")


def _show(args: argparse.Namespace) -> None:
    project = _resolve_project(args.project)
    manifest = _owned(str(project["id"]), args.usecase_id)
    if args.json:
        _print(manifest, True)
        return
    print(f"Use case : {manifest['name']} [{manifest['id']}]")
    print(f"Template : {manifest['template']}")
    print(f"Examples : {manifest.get('examples', 0)} · train {manifest.get('train_rows', 0)} · "
          f"held out {manifest.get('heldout', 0)}")
    print(f"Dataset  : {manifest.get('dataset', {}).get('id', '—')}")
    print(f"Suite    : {manifest.get('suite', {}).get('path', '—')}")
    evaluations = manifest.get("evaluations") or []
    if evaluations:
        print("Evaluations:")
        for evaluation_row in evaluations:
            print(f"  {evaluation_row.get('group_id')} · run {evaluation_row.get('run_id')} · "
                  f"use `fts usecases report {manifest['id']} {evaluation_row.get('group_id')} "
                  f"--project {project['id']}`")


def _delete(args: argparse.Namespace) -> None:
    project = _resolve_project(args.project)
    manifest = _owned(str(project["id"]), args.usecase_id)
    if not args.yes:
        prompt = (f"Delete use case {manifest.get('name')!r}? The held-out suite and result records go; "
                  "the training dataset and runs stay. [y/N] ")
        if not sys.stdin.isatty():
            raise UsecaseCommandError("deleting a use case without a terminal requires --yes")
        if args.json:
            print(prompt, file=sys.stderr, end="")
            answer = input()
        else:
            answer = input(prompt)
        if answer.strip().casefold() not in {"y", "yes"}:
            _print({"ok": False, "cancelled": True}, args.json)
            if not args.json:
                print("Cancelled.")
            return
    result = store.delete(str(project["id"]), args.usecase_id)
    result["deleted"] = {"id": args.usecase_id, "name": manifest.get("name")}
    if args.json:
        _print(result, True)
    else:
        print(f"Deleted use-case record and held-out suite; training dataset {result.get('kept_dataset')} remains.")


def _evaluate(args: argparse.Namespace) -> None:
    project = _resolve_project(args.project)
    _owned(str(project["id"]), args.usecase_id)
    from finetune_studio.webui.testing_jobs import TestingBusy

    try:
        record = evaluate.start(str(project["id"]), args.usecase_id, args.run_id,
                                judge_provider_id=args.judge_provider.strip())
    except (LookupError, evaluate.EvaluationConflict, TestingBusy) as exc:
        raise UsecaseCommandError(str(exc)) from exc
    result = {"ok": True, "project": {"id": project["id"], "name": project["name"]},
              "usecase_id": args.usecase_id, **record}
    if args.json:
        _print(result, True)
    else:
        print(f"Evaluation queued: {record['group_id']} · run {record['run_id']}")
        print(f"Check progress: fts usecases report {args.usecase_id} {record['group_id']} --project {project['id']}")


def _report(args: argparse.Namespace) -> None:
    project = _resolve_project(args.project)
    _owned(str(project["id"]), args.usecase_id)
    try:
        report = evaluate.report(str(project["id"]), args.usecase_id, args.group_id)
    except LookupError as exc:
        raise UsecaseCommandError(str(exc)) from exc
    if args.json:
        _print(report, True)
        return
    status = report.get("status") or {}
    print(f"{report.get('usecase', {}).get('name', args.usecase_id)} · {report.get('usecase', {}).get('template')}")
    print(f"Run: {status.get('run', 'unknown')} · judge: {status.get('judge') or 'not required'}")
    for side in ("base", "tuned"):
        result = report.get(side) or {}
        scores = result.get("scores") or result.get("metrics") or {}
        print(f"{side.title():<6} {result.get('status') or '—'} · {scores}")
    if report.get("paired"):
        print(f"Paired: {report['paired']}")
    if report.get("ab"):
        print(f"Style A/B: {report['ab']}")
    for note in report.get("notes") or []:
        print(f"Note: {note}")


def _ab(args: argparse.Namespace) -> None:
    project = _resolve_project(args.project)
    _owned(str(project["id"]), args.usecase_id)
    from finetune_studio.testing.judge import JudgeUnavailable
    from finetune_studio.webui.testing_jobs import TestingBusy

    try:
        state = evaluate.start_ab(str(project["id"]), args.usecase_id, args.group_id, args.judge_provider.strip())
    except (LookupError, ValueError, evaluate.EvaluationConflict, TestingBusy, JudgeUnavailable) as exc:
        raise UsecaseCommandError(str(exc)) from exc
    result = {"ok": True, "usecase_id": args.usecase_id, "group_id": args.group_id, **state}
    if args.json:
        _print(result, True)
    else:
        print(f"Style A/B judge {state.get('status')}: {args.group_id} · {state.get('done', 0)}/{state.get('total', 0)}")


def cmd_usecases(args: argparse.Namespace) -> None:
    """Handle ``fts usecases {templates,build,list,show,delete,evaluate,report,ab}``."""
    db.init_db()
    actions = {"templates": _templates, "build": _build, "list": _list, "show": _show, "delete": _delete,
               "evaluate": _evaluate, "report": _report, "ab": _ab}
    try:
        actions[args.usecases_action](args)
    except (UsecaseCommandError, rows.RowsError, json.JSONDecodeError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        raise SystemExit(1) from exc
