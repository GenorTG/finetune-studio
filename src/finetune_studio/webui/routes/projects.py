"""Projects API — list/create/select Project and manage Training Runs.

WHY THIS EXISTS
===============
The Studio is organised around Projects. Each Project:
  - has a base model + system prompt
  - owns N RAGs (legal, social, paperwork, ...)
  - owns N Training Runs (each with settings + metrics + benchmark results)
  - has one "production" Run that the Inference Chat loads

This route file covers projects and run records (RAG corpora live in
routes/rag.py + routes/project_rag.py).
"""

from __future__ import annotations

import logging
import re
import time

from fastapi import APIRouter, Request
from fastapi.responses import JSONResponse

from finetune_studio import db

log = logging.getLogger(__name__)

router = APIRouter()


def _project_404(pid: str) -> JSONResponse | None:
    """Return a 404 response when the project does not exist, else None.

    This module answers errors with ``JSONResponse`` (see ``get_project``),
    so the guard matches that style instead of raising HTTPException. Called
    before any DB read or filesystem scan so a bad pid starts no work.
    """
    if not db.get_project(pid):
        return JSONResponse({"error": "project not found"}, status_code=404)
    return None


def _get_owned_run(pid: str, rid: str) -> tuple[dict | None, JSONResponse | None]:
    """Fetch a run and verify it belongs to ``pid``.

    Returns ``(run, None)`` on success or ``(None, 404 response)`` when the
    run is missing or belongs to a different project — a run id from another
    project must never be readable/mutable through this project's URL.
    """
    run = db.get_run(rid)
    if not run or (run.get("project_id") and run.get("project_id") != pid):
        return None, JSONResponse({"error": "run not found"}, status_code=404)
    return run, None


# ── Projects ─────────────────────────────────────────────────────────────

@router.get("")
async def list_projects():
    return db.list_projects()


@router.post("")
async def create_project(request: Request):
    body = await request.json()
    p = db.create_project(
        name=body.get("name", "Untitled"),
        description=body.get("description", ""),
        base_model=body.get("base_model", ""),
        system_prompt=body.get("system_prompt", ""),
    )
    return p


@router.get("/{pid}")
async def get_project(pid: str):
    missing = _project_404(pid)
    if missing is not None:
        return missing
    p = db.get_project(pid)
    p["rags"] = db.list_rags(pid)
    p["runs"] = db.list_runs(pid)
    # Attach model exports found on disk for each run.
    from finetune_studio.webui.routes.pages import _scan_run_models
    p["models"] = _scan_run_models(p["runs"])
    # Attach benchmark summaries to runs.
    for run in p["runs"]:
        run["benchmarks"] = db.list_benchmarks(run["id"])
    return p


@router.patch("/{pid}")
async def update_project(pid: str, request: Request):
    missing = _project_404(pid)
    if missing is not None:
        return missing
    body = await request.json()
    return db.update_project(pid, **body)


@router.delete("/{pid}")
async def delete_project(pid: str):
    """Delete a project's DB rows and every on-disk tree that stores its data.

    Audit fix (2026-10-01): this previously only called ``db.delete_project``
    — a project's file library/QA pairs/logs (``~/.finetune-studio/projects/
    <pid>/``), its datasets (``data/projects/<pid>/``), and every training
    run's merged weights + GGUF exports (``output/projects/<pid>/``) were
    left on disk forever. The DB row disappeared so the project vanished
    from the UI, but the delete call had claimed success while silently
    leaving the actual data behind — found while a 121GB cleanup turned up
    three "deleted" projects whose files were all still resident.
    """
    missing = _project_404(pid)
    if missing is not None:
        return missing
    db.delete_project(pid)

    import shutil
    from pathlib import Path

    from finetune_studio.config import settings
    from finetune_studio.data.fs.paths import rag_corpus_dir
    from finetune_studio.data.fs.paths import root as fts_root

    for project_dir in (
        fts_root() / "projects" / pid,
        Path(settings.db_path).parent / "projects" / pid,
        Path("output") / "projects" / pid,
        rag_corpus_dir(pid),
    ):
        shutil.rmtree(project_dir, ignore_errors=True)

    return {"ok": True}


ARCHIVE_VERSION = "2.0"
_PROJECT_META_FIELDS = ("name", "description", "base_model", "system_prompt", "tags", "notes")


def _projects_root():
    from finetune_studio.data.fs.paths import root as fts_root

    return fts_root() / "projects"


def _corpora_root():
    from finetune_studio.data.fs.paths import rag_corpora_root

    return rag_corpora_root()


def _safe_member_parts(name: str) -> tuple[str, ...] | None:
    """Split an archive member name; None if absolute or containing ``..``."""
    norm = name.replace("\\", "/")
    if norm.startswith("/") or "\x00" in norm:
        return None
    parts = tuple(p for p in norm.split("/") if p not in ("", "."))
    if not parts or ".." in parts or ":" in parts[0]:
        return None
    return parts


@router.get("/{pid}/export")
async def export_project(pid: str, name: str | None = None, fmt: str = "tar.gz"):
    """Export one project as a self-contained archive.

    Layout: ``manifest.json`` (project metadata), ``projects/<pid>/...`` and
    ``rag_corpora/<pid>/...`` — exactly what ``import_project`` reads back.
    """
    import io
    import json
    import tarfile

    from fastapi.responses import StreamingResponse

    from finetune_studio.data.fs.archive_library import (
        LIBRARY_MEMBER,
        dump_file_library,
    )

    project = db.get_project(pid)
    if not project:
        return JSONResponse({"error": "project not found"}, status_code=404)
    if fmt not in ("tar.gz", "tar"):
        return JSONResponse({"error": "fmt must be tar.gz or tar"}, status_code=400)
    if _safe_member_parts(pid) is None or len(_safe_member_parts(pid) or ()) != 1:
        return JSONResponse({"error": "invalid project id"}, status_code=400)

    def stream():
        buf = io.BytesIO()
        mode = "w:gz" if fmt == "tar.gz" else "w"
        with tarfile.open(fileobj=buf, mode=mode) as tar:
            data_dir = _projects_root() / pid
            if data_dir.exists():
                tar.add(str(data_dir), arcname=f"projects/{pid}")
            rag_dir = _corpora_root() / pid
            if rag_dir.exists():
                tar.add(str(rag_dir), arcname=f"rag_corpora/{pid}")
            manifest = json.dumps({
                "project_id": pid,
                "exported_at": time.time(),
                "version": ARCHIVE_VERSION,
                "project": {k: project.get(k) or "" for k in _PROJECT_META_FIELDS},
            }, indent=2)
            manifest_bytes = manifest.encode()
            info = tarfile.TarInfo(name="manifest.json")
            info.size = len(manifest_bytes)
            tar.addfile(info, io.BytesIO(manifest_bytes))
            lib_bytes = json.dumps(dump_file_library(pid)).encode()
            info = tarfile.TarInfo(name=LIBRARY_MEMBER)
            info.size = len(lib_bytes)
            tar.addfile(info, io.BytesIO(lib_bytes))
        buf.seek(0)
        yield buf.read()

    safe_name = re.sub(r"[^\w.\-]", "_", name or f"project-{pid}")
    filename = f"{safe_name}.{fmt}"
    media = "application/gzip" if fmt == "tar.gz" else "application/x-tar"
    return StreamingResponse(stream(), media_type=media,
        headers={"Content-Disposition": f"attachment; filename={filename}"})


@router.post("/import")
async def import_project(request: Request):
    """Import ONE project from an archive produced by ``export_project``.

    Nothing is extracted until every member is validated: only regular files
    and directories under ``projects/<old_id>/`` or ``rag_corpora/<old_id>/``
    are accepted (no links, devices, absolute or ``..`` paths). Files land
    under a freshly generated project id, never the archived one.
    """
    import io
    import json
    import shutil
    import tarfile

    from finetune_studio.data.fs.archive_library import (
        LIBRARY_MEMBER,
        rebase_project_files,
        restore_file_library,
    )

    form = await request.form()
    file = form.get("file")
    if not file:
        return JSONResponse({"error": "no file"}, status_code=400)

    buf = io.BytesIO(await file.read())

    def bad(msg: str):
        return JSONResponse({"error": f"invalid archive: {msg}"}, status_code=400)

    try:
        with tarfile.open(fileobj=buf, mode="r:*") as tar:
            manifest: dict = {}
            library: object = None
            plan: list[tuple[tarfile.TarInfo, str, tuple[str, ...]]] = []
            old_ids: set[str] = set()
            for member in tar.getmembers():
                parts = _safe_member_parts(member.name)
                if parts is None:
                    return bad(f"unsafe member path {member.name!r}")
                if parts == ("manifest.json",):
                    f = tar.extractfile(member) if member.isfile() else None
                    if f is None:
                        return bad("manifest.json is not a file")
                    try:
                        manifest = json.loads(f.read().decode())
                    except (UnicodeDecodeError, json.JSONDecodeError):
                        return bad("manifest.json is not valid JSON")
                    if not isinstance(manifest, dict):
                        return bad("manifest.json must be an object")
                    continue
                if parts == (LIBRARY_MEMBER,):
                    f = tar.extractfile(member) if member.isfile() else None
                    if f is None:
                        return bad("file_library.json is not a file")
                    try:
                        library = json.loads(f.read().decode())
                    except (UnicodeDecodeError, json.JSONDecodeError):
                        return bad("file_library.json is not valid JSON")
                    continue
                if parts[0] not in ("projects", "rag_corpora"):
                    return bad(f"unexpected member {member.name!r}")
                if len(parts) == 1:
                    continue
                if not (member.isfile() or member.isdir()):
                    return bad(f"unsupported member type: {member.name!r}")
                old_ids.add(parts[1])
                plan.append((member, parts[0], parts[1:]))

            declared = manifest.get("project_id")
            if declared is not None:
                if not isinstance(declared, str) or _safe_member_parts(declared) != (declared,):
                    return bad("manifest project_id is invalid")
                if old_ids - {declared}:
                    return bad("archive contains data for other projects")
                old_id = declared
            elif len(old_ids) == 1:
                old_id = next(iter(old_ids))
            else:
                return bad("expected exactly one project")
            if not any(root == "projects" for _m, root, _p in plan):
                return bad("no projects dir")

            meta = manifest.get("project") if isinstance(manifest.get("project"), dict) else {}
            meta = {k: str(meta.get(k) or "") for k in _PROJECT_META_FIELDS}
            proj_name = meta["name"] or f"Imported {old_id[:8]}"
            new_proj = db.create_project(
                name=proj_name,
                description=meta["description"] or "Imported from archive",
                base_model=meta["base_model"],
                system_prompt=meta["system_prompt"] or "You are a helpful assistant.",
            )
            new_id = new_proj["id"]
            db.update_project(new_id, tags=meta["tags"] or "imported", notes=meta["notes"])

            dests = {"projects": _projects_root() / new_id, "rag_corpora": _corpora_root() / new_id}
            for member, root, parts in plan:
                dest = dests[root].joinpath(*parts[1:])
                if member.isdir():
                    dest.mkdir(parents=True, exist_ok=True)
                    continue
                dest.parent.mkdir(parents=True, exist_ok=True)
                src = tar.extractfile(member)
                if src is None:
                    continue
                with src, dest.open("wb") as out:
                    shutil.copyfileobj(src, out)
            dests["projects"].mkdir(parents=True, exist_ok=True)
            rebase_project_files(dests["projects"], old_id)
            restore_file_library(library, old_id, new_id, dests["projects"])
            return {"ok": True, "imported": [{"old_id": old_id, "new_id": new_id, "name": proj_name}],
                    "count": 1}
    except (tarfile.TarError, EOFError) as e:
        return bad(str(e))


@router.post("/{pid}/promote")
async def promote_run(pid: str, request: Request):
    """Set a Training Run as the Project's production model."""
    body = await request.json()
    run_id = body.get("run_id", "")
    run, err = _get_owned_run(pid, run_id)
    if err is not None:
        return err
    db.update_project(pid, production_run=run_id)
    return {"ok": True, "run": run}


# ── Training Runs ────────────────────────────────────────────────────────

@router.get("/{pid}/runs")
async def list_runs(pid: str):
    missing = _project_404(pid)
    if missing is not None:
        return missing
    return db.list_runs(pid)


@router.post("/{pid}/runs")
async def create_run(pid: str, request: Request):
    missing = _project_404(pid)
    if missing is not None:
        return missing
    body = await request.json()
    run = db.create_run(
        project_id=pid,
        name=body.get("name", "Run"),
        base_model=body.get("base_model", ""),
        data_path=body.get("data_path", ""),
        rag_ids=body.get("rag_ids", []),
        settings_obj=body.get("settings", {}),
        system_prompt=body.get("system_prompt", ""),
        parent_run_id=body.get("parent_run_id"),
        notes=body.get("notes", ""),
    )
    return run


@router.get("/{pid}/runs/{rid}")
async def get_run(pid: str, rid: str):
    run, err = _get_owned_run(pid, rid)
    if err is not None:
        return err
    run["benchmarks"] = db.list_benchmarks(rid)
    return run


@router.patch("/{pid}/runs/{rid}")
async def update_run(pid: str, rid: str, request: Request):
    _run, err = _get_owned_run(pid, rid)
    if err is not None:
        return err
    body = await request.json()
    return db.update_run(rid, **body)


@router.delete("/{pid}/runs/{rid}")
async def delete_run(pid: str, rid: str):
    _run, err = _get_owned_run(pid, rid)
    if err is not None:
        return err
    db.delete_run(rid)
    return {"ok": True}
