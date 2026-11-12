"""Project versioning routes: CRUD + subset-dataset builds.

API surface (all JSON, curl-first; the WebUI guided flow calls these):
- GET/POST /projects/{pid}/versions           — list / save snapshot
- GET    /projects/{pid}/versions/{vid}       — one version (manifest decoded)
- GET    /projects/{pid}/versions/{vid}/lineage — parent walk to root
- DELETE /projects/{pid}/versions/{vid}       — manifest-only delete
- POST   /projects/{pid}/datasets/subset      — hand-picked sources → registered dataset
"""
from __future__ import annotations

import json
import logging
import re
import time
from typing import Any

from fastapi import APIRouter, Request
from fastapi.responses import JSONResponse

from finetune_studio import db
from finetune_studio.db.datasets import (
    count_qa_pairs,
    create_dataset,
    datasets_dir,
    get_dataset_by_path,
)
from finetune_studio.project_versioning import ProjectVersionError, save_project_version

router = APIRouter()
log = logging.getLogger("finetune_studio.versions")


def _manifest_of(v: dict) -> dict[str, Any]:
    try:
        return json.loads(v.get("manifest_json") or "{}")
    except Exception:  # noqa: BLE001
        return {}


@router.get("/projects/{pid}/versions")
async def list_versions(pid: str):
    if not db.get_project(pid):
        return JSONResponse({"detail": "project not found"}, status_code=404)
    return {"versions": db.list_versions(pid)}


@router.post("/projects/{pid}/versions")
async def save_version(pid: str, request: Request):
    try:
        body = await request.json()
    except ValueError:
        return JSONResponse({"detail": "invalid JSON"}, status_code=400)
    if not isinstance(body, dict):
        return JSONResponse({"detail": "expected JSON object"}, status_code=400)
    try:
        return save_project_version(pid, body)
    except ProjectVersionError as exc:
        return JSONResponse({"detail": exc.detail}, status_code=exc.status_code)


@router.get("/projects/{pid}/versions/{vid}")
async def get_version(vid: str, pid: str):
    v = db.get_version(vid)
    if not v or v.get("project_id") != pid:
        return JSONResponse({"detail": "not found"}, status_code=404)
    out = dict(v)
    out["manifest"] = _manifest_of(out)
    return out


@router.get("/projects/{pid}/versions/{vid}/lineage")
async def get_lineage(vid: str, pid: str):
    if not db.get_project(pid):
        return JSONResponse({"detail": "project not found"}, status_code=404)
    chain = db.version_lineage(pid, vid)
    if not chain:
        return JSONResponse({"detail": "not found"}, status_code=404)
    return {"lineage": [{"id": v["id"], "version_number": v["version_number"],
                         "label": v["label"], "created_at": v["created_at"]} for v in chain]}


@router.delete("/projects/{pid}/versions/{vid}")
async def delete_version(vid: str, pid: str):
    v = db.get_version(vid)
    if not v or v.get("project_id") != pid:
        return JSONResponse({"detail": "not found"}, status_code=404)
    return {"ok": db.delete_version(vid)}


@router.post("/projects/{pid}/datasets/subset")
async def build_subset_dataset(pid: str, request: Request):
    """Specialized build: hand-picked sources → completeness gate → registered dataset.

    Body: {source_ids: [...], fmt: sharegpt, name: optional, force: optional bool,
           grounded_share: optional 0-1 (default: auto), distractors: optional 0-2}
    Same contract as the full export, scoped to the picked files: every one must be complete (each data point of its
    inventory stated by an approved pair or excluded), else 409 with the incomplete files unless ``force``.
    """
    from finetune_studio.data.mining import project_coverage
    from finetune_studio.data.prep.export import export_qa_jsonl_from_sources
    from finetune_studio.data.prep.grounding import resolve_grounding
    from finetune_studio.knowledge.sources import names as source_names

    proj = db.get_project(pid)
    if not proj:
        return JSONResponse({"detail": "project not found"}, status_code=404)
    body = await request.json()
    source_ids = [s for s in (body.get("source_ids") or []) if s]
    if not source_ids:
        return JSONResponse({"detail": "source_ids required"}, status_code=400)

    names = source_names(pid)
    known = set(names)
    missing = [s for s in source_ids if s not in known]
    if missing:
        return JSONResponse(
            {"detail": f"unknown source ids ({len(missing)}): {missing[:10]}"}, status_code=400)

    # Completeness gate BEFORE export, scoped to the picked files (mirror of the full-export gate).
    incomplete = [{"source_id": c.source_id, "filename": names.get(c.source_id), "status": c.status, "counts": c.counts}
                  for c in project_coverage(pid, source_ids) if c.status != "complete"]
    if incomplete and body.get("force") is not True:
        return JSONResponse({"detail": "dataset build blocked: not every data point of the selected files is approved yet",
                             "incomplete_files": incomplete, "hint": "send force: true to build anyway"}, status_code=409)

    fmt = body.get("fmt", "sharegpt")
    try:
        # Same grounded-rows policy as the full export (body: grounded_share, distractors).
        grounding = resolve_grounding(pid, body.get("grounded_share"), int(body.get("distractors") or 0))
        payload = export_qa_jsonl_from_sources(pid, source_ids, fmt=fmt, only="approved",
                                               grounding=grounding)
    except (ValueError, TypeError) as e:
        return JSONResponse({"detail": str(e)}, status_code=400)
    if not payload.strip():
        return JSONResponse({"detail": "no approved pairs for the selected sources"}, status_code=400)

    ds_dir = datasets_dir(pid)
    ts = time.strftime("%Y%m%d-%H%M%S")
    base_name = re.sub(r"[^\w.\-]", "_", str(body.get("name") or "")).strip("._") \
        or f"subset-{len(source_ids)}src"
    target = ds_dir / f"{pid}-{base_name}-{ts}.jsonl"
    target.write_text(payload, encoding="utf-8")
    existing = get_dataset_by_path(pid, str(target))
    if existing:
        ds = existing
        db.update_dataset(ds["id"], qa_count=count_qa_pairs(str(target)),
                          size_bytes=target.stat().st_size)
    else:
        ds = create_dataset(project_id=pid, name=target.stem, data_path=str(target),
                            source="subset-picked", qa_count=count_qa_pairs(str(target)),
                            size_bytes=target.stat().st_size)
    # Per-source honesty: how many pairs came from each picked source.
    per_source = {s: 0 for s in source_ids}
    for line in payload.splitlines():
        if not line.strip():
            continue
        try:
            per_source[json.loads(line).get("source_id", "")] += 1
        except (json.JSONDecodeError, KeyError, TypeError) as exc:
            log.warning("subset per-source tally skipped a line: %s", exc)
    return {"dataset": ds, "path": str(target), "rows": count_qa_pairs(str(target)),
            "per_source": per_source, "incomplete_files": incomplete}
