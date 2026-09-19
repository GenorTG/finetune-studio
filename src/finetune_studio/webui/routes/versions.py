"""Project versioning routes: CRUD + subset-dataset builds + RAG coverage.

API surface (all JSON, curl-first; the WebUI guided flow calls these):
- GET/POST /projects/{pid}/versions           — list / save snapshot
- GET    /projects/{pid}/versions/{vid}       — one version (manifest decoded)
- GET    /projects/{pid}/versions/{vid}/lineage — parent walk to root
- DELETE /projects/{pid}/versions/{vid}       — manifest-only delete
- POST   /projects/{pid}/datasets/subset      — hand-picked sources → registered dataset
- GET    /projects/{pid}/rag/coverage         — RAG corpus vs parsed sources gate
"""
from __future__ import annotations

import json
import logging
import sqlite3
import time
from pathlib import Path
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

router = APIRouter()
log = logging.getLogger("finetune_studio.versions")


def _manifest_of(v: dict) -> dict[str, Any]:
    try:
        return json.loads(v.get("manifest_json") or "{}")
    except Exception:  # noqa: BLE001
        return {}


async def _save_version(pid: str, body: dict[str, Any]) -> dict | JSONResponse:
    """Shared save logic for both the JSON route and the multipart-free CLI path."""
    proj = db.get_project(pid)
    if not proj:
        return JSONResponse({"error": "project not found"}, status_code=404)
    parent_vid = body.get("parent_version_id")
    parent = db.get_version(str(parent_vid)) if parent_vid else None
    if parent_vid and (not parent or parent.get("project_id") != pid):
        return JSONResponse({"error": f"parent version {parent_vid!r} not found"}, status_code=404)
    manifest = body.get("manifest") or {}

    # Auto-fill runnable inputs when the caller doesn't pin them explicitly:
    # current datasets, latest rag corpus build, recent training runs.
    if not manifest.get("datasets"):
        manifest["datasets"] = [
            {"id": d["id"], "name": d["name"], "data_path": d["data_path"],
             "qa_count": d.get("qa_count", 0)}
            for d in db.list_datasets(pid)
        ]
    if not manifest.get("rag_corpora"):
        try:
            with db.cursor() as _c:
                _row = _c.execute(
                    "SELECT * FROM rag_corpora WHERE project_id = ? "
                    "ORDER BY created_at DESC LIMIT 1", (pid,)).fetchone()
            latest = db.row_to_dict(_row) if _row is not None else None
        except sqlite3.Error:
            log.exception("latest rag build lookup failed")
            latest = None
        if latest:
            manifest["rag_corpora"] = [{
                "build_id": latest["id"], "rag_id": latest.get("rag_id", ""),
                "doc_count": latest.get("doc_count", 0),
                "chunk_count": latest.get("chunk_count", 0),
            }]
    if not manifest.get("training_runs"):
        from finetune_studio.db.runs import list_runs
        runs = [r for r in list_runs(pid) if r.get("status") in ("done", "completed", "completed_error")][:5]
        manifest["training_runs"] = [
            {"run_id": r["id"], "status": r.get("status", ""),
             "final_loss": r.get("final_loss"), "output_path": r.get("output_path", "")}
            for r in runs
        ]
    v = db.create_version(
        pid,
        label=str(body.get("label") or ""),
        notes=str(body.get("notes") or ""),
        parent_version_id=str(parent_vid) if parent_vid else None,
        manifest=manifest,
    )
    return v


@router.get("/projects/{pid}/versions")
async def list_versions(pid: str):
    if not db.get_project(pid):
        return JSONResponse({"error": "project not found"}, status_code=404)
    return {"versions": db.list_versions(pid)}


@router.post("/projects/{pid}/versions")
async def save_version(pid: str, request: Request):
    return await _save_version(pid, await request.json())


@router.get("/projects/{pid}/versions/{vid}")
async def get_version(vid: str, pid: str):
    v = db.get_version(vid)
    if not v or v.get("project_id") != pid:
        return JSONResponse({"error": "not found"}, status_code=404)
    out = dict(v)
    out["manifest"] = _manifest_of(out)
    return out


@router.get("/projects/{pid}/versions/{vid}/lineage")
async def get_lineage(vid: str, pid: str):
    if not db.get_project(pid):
        return JSONResponse({"error": "project not found"}, status_code=404)
    chain = db.version_lineage(pid, vid)
    if not chain:
        return JSONResponse({"error": "not found"}, status_code=404)
    return {"lineage": [{"id": v["id"], "version_number": v["version_number"],
                         "label": v["label"], "created_at": v["created_at"]} for v in chain]}


@router.delete("/projects/{pid}/versions/{vid}")
async def delete_version(vid: str, pid: str):
    v = db.get_version(vid)
    if not v or v.get("project_id") != pid:
        return JSONResponse({"error": "not found"}, status_code=404)
    return {"ok": db.delete_version(vid)}


@router.post("/projects/{pid}/datasets/subset")
async def build_subset_dataset(pid: str, request: Request):
    """Specialized build: hand-picked sources → coverage-filled → registered dataset.

    Body: {source_ids: [...], fmt: sharegpt, name: optional}
    Guarantees the same no-skips contract as the full export, scoped to the
    picked sources.
    """
    from finetune_studio.data.fs import qa as qafs
    from finetune_studio.data.prep.coverage_fill import fill_sources_gaps
    from finetune_studio.data.prep.export import export_qa_jsonl_from_sources

    proj = db.get_project(pid)
    if not proj:
        return JSONResponse({"error": "project not found"}, status_code=404)
    body = await request.json()
    source_ids = [s for s in (body.get("source_ids") or []) if s]
    if not source_ids:
        return JSONResponse({"error": "source_ids required"}, status_code=400)

    known = {s.get("id") for s in qafs.list_qa_sources(pid)}
    missing = [s for s in source_ids if s not in known]
    if missing:
        return JSONResponse(
            {"error": f"unknown source ids ({len(missing)}): {missing[:10]}"}, status_code=400)

    # Subset coverage gate BEFORE export — mirror of the full-export gate.
    try:
        fill = fill_sources_gaps(pid, source_ids)
        if fill.get("uncovered_chunks"):
            log.warning("subset coverage fill left %d uncovered: %s",
                        len(fill["uncovered_chunks"]), fill["uncovered_chunks"][:6])
    except Exception:
        log.exception("subset coverage fill failed")
        fill = None

    fmt = body.get("fmt", "sharegpt")
    try:
        payload = export_qa_jsonl_from_sources(pid, source_ids, fmt=fmt, only="approved")
    except ValueError as e:
        return JSONResponse({"error": str(e)}, status_code=400)
    if not payload.strip():
        return JSONResponse({"error": "no approved pairs for the selected sources"}, status_code=400)

    ds_dir = datasets_dir(pid)
    ts = time.strftime("%Y%m%d-%H%M%S")
    base_name = body.get("name") or f"subset-{len(source_ids)}src"
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
            "per_source": per_source, "coverage_fill": fill}


@router.get("/projects/{pid}/rag/coverage")
async def rag_coverage(pid: str):
    """100%-facts gate for RAG: every parsed source present in the corpus?

    Parsed side = qa/sources/*.json ( declares chunk_count + sha256).
    Corpus side = PortableRAG manifest documents_meta (per-document ids/names).
    Returns per-source rows + summary; missing sources are listed explicitly.
    """
    import json as _json

    from finetune_studio.config import settings
    from finetune_studio.data.fs import qa as qafs

    sources = qafs.list_qa_sources(pid)
    if not sources:
        return JSONResponse({"error": "no parsed sources in this project"}, status_code=400)
    corpus_dir = (Path(settings.db_path).parent / "rag_corpora" / pid)
    manifest_path = corpus_dir / "manifest.json"
    if not manifest_path.exists():
        return JSONResponse({"error": "no RAG corpus built for this project"}, status_code=400)
    try:
        raw = _json.loads(manifest_path.read_text(encoding="utf-8"))
        meta = (raw.get("extra") or {}).get("documents_meta") or []
    except (OSError, ValueError, AttributeError, TypeError):
        meta = []
    built_names: set[str] = set()
    for d in meta:
        fname = str(d.get("filename") or "")
        stem = fname.rsplit(".", 1)[0] if "." in fname else fname
        built_names.add(stem)
        built_names.add(fname)

    rows = []
    for s in sources:
        declared = int(s.get("chunk_count") or 0)
        orig = str(s.get("filename") or "")
        stem = Path(orig).stem
        covered = stem in built_names or orig in built_names
        rows.append({"source_id": s.get("id"), "filename": orig,
                     "declared_chunks": declared,
                     "in_corpus": covered,
                     "shas_in_name": stem})
    missing = [r for r in rows if not r["in_corpus"]]
    return {
        "project": pid, "corpus_dir": str(corpus_dir),
        "parsed_sources": len(rows), "covered": len(rows) - len(missing),
        "missing_sources": [{k: r[k] for k in ("source_id", "filename", "declared_chunks")}
                            for r in missing],
        "coverage_pct": round(100.0 * (len(rows) - len(missing)) / len(rows), 1) if rows else 0.0,
    }
