"""`fts dataset build` — build + register a training dataset from a project's Q&A pairs.

Same pipeline as the WebUI export (``data.prep.dataset_build``): coverage-fill
gate → dedupe → optional grounded rows → JSONL in the project's datasets dir,
registered so the Training tab sees it. Works against the local DB/project
files; no running server needed.

Also `fts dataset build-preference`: author DPO preference pairs (chosen vs rejected answers) from the
approved pairs; loads the helper model itself when none is resident and unloads it afterwards.

Examples:
  fts dataset build --project my-docs
  fts dataset build --project my-docs --grounded-share 0.5 --distractors 1 --seed 7
  fts dataset build --project my-docs --include-unreviewed-fill
  fts dataset build --project my-docs --no-rag-grounding --fmt openai --out exports/plain.jsonl
  fts dataset build-preference --project my-docs --kinds hallucination,abstain --max-pairs 60 --json
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

from fastapi import HTTPException

from finetune_studio import db
from finetune_studio.data.prep import preference as pref
from finetune_studio.data.prep.dataset_build import (
    BuiltDataset,
    CoverageCheckFailed,
    ExportBlocked,
    build_project_dataset,
)
from finetune_studio.data.prep.grounding import GroundingOptions, resolve_grounding

_UNCOVERED_SHOWN = 10


class DatasetBuildError(Exception):
    """A user-facing failure: printed to stderr, exit status 1."""


def resolve_project(ref: str) -> dict[str, Any]:
    """Project row for an id or an exact (then case-insensitive) name; error on none/ambiguous."""
    ref = (ref or "").strip()
    if not ref:
        raise DatasetBuildError("--project is required (project id or name)")
    by_id = db.get_project(ref)
    if by_id:
        return by_id
    projects = db.list_projects()
    for match in (lambda n: n == ref, lambda n: n.casefold() == ref.casefold()):
        hits = [p for p in projects if match(str(p.get("name") or ""))]
        if len(hits) == 1:
            return hits[0]
        if len(hits) > 1:
            ids = ", ".join(str(p["id"]) for p in hits)
            raise DatasetBuildError(f"project name {ref!r} is ambiguous; use one of these ids: {ids}")
    raise DatasetBuildError(f"project not found: {ref!r}")


def grounding_from_args(pid: str, args: argparse.Namespace) -> GroundingOptions | None:
    """Same policy as the route: omitted share = auto (on when a RAG corpus is built), 0 = off."""
    share = 0.0 if args.no_rag_grounding else args.grounded_share
    try:
        return resolve_grounding(pid, share, args.distractors, args.seed)
    except ValueError as exc:
        raise DatasetBuildError(str(exc)) from exc


def _resolve_out(pid: str, out: str) -> Path:
    from finetune_studio.data.fs.paths import resolve_in_project
    try:
        return resolve_in_project(pid, out, what="--out")
    except HTTPException as exc:
        raise DatasetBuildError(f"--out: {exc.detail}") from exc


def _summary(built: BuiltDataset, project: dict[str, Any], fmt: str, out_copy: Path | None) -> dict[str, Any]:
    stats = built.export.grounding
    cov = built.coverage
    return {
        "project": {"id": project["id"], "name": project.get("name")},
        "dataset": {"id": built.persisted.dataset.get("id"), "name": built.persisted.dataset.get("name"),
                    "path": str(built.persisted.path), "format": fmt},
        "rows": built.export.rows,
        "grounded_rows": built.grounded_rows,
        "grounding": stats.as_dict() if stats else None,
        "coverage_fill": {"mode": cov.get("mode"),
                          "pairs_created": cov.get("pairs_created", 0),
                          "pairs_promoted": cov.get("pairs_promoted", 0),
                          "chunks_filled": cov.get("chunks_filled", 0),
                          "skipped_no_content": cov.get("skipped_no_content", 0),
                          "uncovered": len(cov.get("uncovered_chunks") or [])},
        "copy": str(out_copy) if out_copy else None,
    }


def _print_summary(s: dict[str, Any]) -> None:
    ds, cf = s["dataset"], s["coverage_fill"]
    print(f"Dataset  : {ds['name']}  [{ds['format']}]")
    print(f"File     : {ds['path']}")
    if s["copy"]:
        print(f"Copy     : {s['copy']}")
    print(f"Rows     : {s['rows']} ({s['grounded_rows']} grounded with retrieved context)")
    added = cf["pairs_created"] + cf["pairs_promoted"]
    kept_out = "approved, in this dataset" if cf["mode"] == "approve" else "left pending for review, NOT in this dataset"
    print(f"Coverage : {cf['chunks_filled']} chunk(s) filled with {added} extractive pair(s) ({kept_out}); "
          f"{cf['uncovered']} still uncovered")


def _build(args: argparse.Namespace) -> None:
    db.init_db()  # idempotent, additive — a fresh FTS_ROOT has no schema yet
    project = resolve_project(args.project)
    pid = str(project["id"])
    grounding = grounding_from_args(pid, args)
    out_copy = _resolve_out(pid, args.out) if args.out else None
    try:
        built = build_project_dataset(pid, args.fmt, args.only, grounding=grounding,
                                      force=args.force, include_unreviewed_fill=args.include_unreviewed_fill,
                                      name=args.name)
    except ExportBlocked as blocked:
        lines = [str(blocked),
                 f"{len(blocked.uncovered_chunks)} chunk(s) uncovered in: {', '.join(blocked.files)}"]
        for u in blocked.uncovered_chunks[:_UNCOVERED_SHOWN]:
            lines.append(f"  - {u.get('filename') or u.get('source') or '?'} chunk {u.get('chunk_idx', '?')}: "
                         f"{u.get('reason') or u.get('error') or 'no usable Q&A'}")
        lines.append("Pass --force to export anyway with those chunks missing.")
        raise DatasetBuildError("\n".join(lines)) from blocked
    except (CoverageCheckFailed, ValueError) as exc:
        raise DatasetBuildError(str(exc)) from exc
    if out_copy is not None:
        out_copy.parent.mkdir(parents=True, exist_ok=True)
        out_copy.write_text(built.export.body, encoding="utf-8")
    summary = _summary(built, project, args.fmt, out_copy)
    if args.json:
        print(json.dumps(summary, indent=2))
    else:
        _print_summary(summary)


def _parse_kinds(raw: str) -> tuple[str, ...]:
    kinds = tuple(k.strip() for k in raw.split(",") if k.strip())
    bad = [k for k in kinds if k not in pref.KINDS]
    if bad or not kinds:
        raise DatasetBuildError(f"--kinds must be a comma-separated subset of {', '.join(pref.KINDS)}"
                                + (f"; unknown: {', '.join(bad)}" if bad else ""))
    return kinds


def _print_progress(update: dict[str, Any]) -> None:
    kept = ", ".join(f"{k} {n}" for k, n in update["kept"].items())
    print(f"\r  {update['attempted']}/{update['planned']} candidates tried · kept {kept}   ",
          end="", file=sys.stderr, flush=True)


def _print_preference_summary(s: dict[str, Any]) -> None:
    ds, ln = s["dataset"], s["length"]
    print(f"Dataset  : {ds['name']}")
    print(f"File     : {ds['path']}")
    print(f"Pairs    : {s['pairs']} ({', '.join(f'{k} {n}' for k, n in s['by_kind'].items())}); "
          f"split preview {s['split']['train']} train / {s['split']['val']} validation")
    for kind, drops in s["dropped"].items():
        if drops:
            print(f"Dropped  : {kind}: " + ", ".join(f"{r} {n}" for r, n in sorted(drops.items())))
    print(f"Length   : chosen {ln['mean_chosen_chars']} chars vs rejected {ln['mean_rejected_chars']} (ratio {ln['ratio']})")
    if ln["warning"]:
        print(f"Warning  : {ln['warning']}")


def _build_preference(args: argparse.Namespace) -> None:
    db.init_db()
    project = resolve_project(args.project)
    pid = str(project["id"])
    kinds = _parse_kinds(args.kinds)
    try:
        with pref.helper_loaded_for_cli() as generate:
            built = pref.build_preference_dataset(
                pid, kinds, args.max_pairs, args.seed, generate,
                progress=None if args.json else _print_progress)
    except pref.NoUsablePairs as exc:
        raise DatasetBuildError(f"{exc} Dropped: {json.dumps(exc.report.as_dict()['dropped'])}") from exc
    except (pref.PreferenceBuildError, ValueError) as exc:
        raise DatasetBuildError(str(exc)) from exc
    if not args.json:
        print(file=sys.stderr)
    ds = built.persisted.dataset
    summary = {"project": {"id": pid, "name": project.get("name")},
               "dataset": {"id": ds.get("id"), "name": ds.get("name"), "path": str(built.persisted.path)},
               **built.report.as_dict()}
    if args.json:
        print(json.dumps(summary, indent=2))
    else:
        _print_preference_summary(summary)


def cmd_dataset(args: argparse.Namespace) -> None:
    sub = getattr(args, "dataset_command", None)
    if sub not in ("build", "build-preference"):
        print("Usage: fts dataset build|build-preference --project <name|id> [options]  "
              "(see `fts dataset <command> --help`)", file=sys.stderr)
        sys.exit(1)
    try:
        (_build_preference if sub == "build-preference" else _build)(args)
    except DatasetBuildError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        sys.exit(1)
