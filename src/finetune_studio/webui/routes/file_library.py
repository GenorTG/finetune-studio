"""File library routes — Stage 1 of REFACTOR-SPEC.

Endpoints:
  POST   /api/projects/{pid}/files/upload         — single + bulk upload
  GET    /api/projects/{pid}/files                — list files (filterable)
  GET    /api/projects/{pid}/files/trash          — list trashed files
  POST   /api/projects/{pid}/files/trash/purge    — hard-delete old trashed files
  GET    /api/projects/{pid}/files/{fid}          — get one file's metadata
  GET    /api/projects/{pid}/files/{fid}/raw      — download raw bytes (?version=N)
  GET    /api/projects/{pid}/files/{fid}/parsed   — stream parsed markdown preview
  GET    /api/projects/{pid}/files/{fid}/versions — list all raw versions
  GET    /api/projects/{pid}/files/{fid}/conversions — list converted versions
  PATCH  /api/projects/{pid}/files/{fid}/rename   — rename file (DB + disk)
  POST   /api/projects/{pid}/files/{fid}/purge    — hard-delete one trashed file
  POST   /api/projects/{pid}/files/{fid}/move     — move file to folder
  DELETE /api/projects/{pid}/files/{fid}          — soft-delete (moves to trash)
  POST   /api/projects/{pid}/files/{fid}/restore  — restore from trash
  POST   /api/projects/{pid}/files/bulk           — bulk delete/restore/move/reparse/tag
  POST   /api/projects/{pid}/files/download-zip   — zip raw bytes of selected files
  GET    /api/projects/{pid}/files/search-content — substring search inside parsed text
  GET    /api/projects/{pid}/files/{fid}/usage    — where this file went (source/pairs/datasets/runs/RAG)

  POST   /api/projects/{pid}/folders              — create user folder
  GET    /api/projects/{pid}/folders              — list folders (user + auto)
  PATCH  /api/projects/{pid}/folders/{fid}        — rename folder
  DELETE /api/projects/{pid}/folders/{fid}        — delete folder (files survive)

IMPORTANT: FastAPI matches routes in registration order, so the more
specific routes (/files/trash, /files/trash/purge) are declared BEFORE
the generic /files/{fid} catch-all, otherwise the catch-all would eat
them as a fid='trash' lookup.
"""
from __future__ import annotations

import logging

from fastapi import APIRouter, File, Form, HTTPException, Query, Request, UploadFile
from fastapi.responses import FileResponse, JSONResponse, Response

from finetune_studio.data.fs import file_library as fl

log = logging.getLogger(__name__)
router = APIRouter()


# ── Helpers ──────────────────────────────────────────────────────────────

def _project_or_404(pid: str) -> None:
    from finetune_studio import db
    if not db.get_project(pid):
        raise HTTPException(status_code=404, detail="project not found")


# ── File upload (single + bulk) ──────────────────────────────────────────

@router.post("/projects/{pid}/files/upload")
async def upload_files(
    pid: str,
    request: Request,
    files: list[UploadFile] = File(...),  # noqa: B008  # FastAPI requires File() default at def site
    folder_id: str | None = Form(None),
    uploaded_by: str = Form("user"),
):
    """Upload one or more files. Supports both single-file (curl -F file=@x)
    and bulk upload (curl -F files=@x -F files=@y).

    For bulk: each file is processed independently; the response is a
    structured report listing uploaded, duplicate, and error items.

    dedup: sha256 of raw bytes. If an exact-hash match exists in this
    project, the file is reported as 'duplicate' (not written) with
    duplicate_of pointing to the existing file's name.
    """
    _project_or_404(pid)
    fl.ensure_dirs(pid)

    if not files:
        raise HTTPException(status_code=400, detail="no files in request")

    report: list[dict] = []
    counts = {"uploaded": 0, "duplicates_skipped": 0, "errors": 0}

    for up in files:
        name = up.filename or "unnamed"
        try:
            data = await up.read()
        except Exception as e:  # noqa: BLE001
            report.append({
                "filename": name,
                "status": "error",
                "error": f"read failed: {e}",
            })
            counts["errors"] += 1
            continue

        if not data:
            report.append({
                "filename": name,
                "status": "error",
                "error": "empty upload",
            })
            counts["errors"] += 1
            continue

        # Dedup check
        h = fl.sha256_of_bytes(data)
        existing = fl.find_existing_hash(pid, h)
        if existing:
            item: dict = {
                "filename": name,
                "status": "duplicate",
                "duplicate_of": existing["original_name"],
                "duplicate_file_id": existing["id"],
            }
            # Still ensure text duplicates are selectable as parsed sources.
            from finetune_studio.data.fs.qa import maybe_auto_promote_upload
            source = maybe_auto_promote_upload(
                pid,
                existing["id"],
                name,
                mime_type=up.content_type or existing.get("mime_type") or "",
            )
            if source:
                item["source"] = source
                item["source_id"] = source.get("id")
            report.append(item)
            counts["duplicates_skipped"] += 1
            continue

        try:
            meta = fl.write_uploaded_file(
                pid=pid,
                data=data,
                original_name=name,
                mime_hint=up.content_type,
                uploaded_by=uploaded_by,
            )
        except Exception as e:
            log.exception("upload write failed for %s", name)
            report.append({"filename": name, "status": "error", "error": str(e)})
            counts["errors"] += 1
            continue

        # Optionally move to a user folder (if folder_id was provided and is
        # a USER folder, not an auto raw folder).
        if folder_id:
            from finetune_studio import db
            with db.cursor() as c:
                row = c.execute(
                    "SELECT kind FROM file_folders WHERE id = ? AND project_id = ?",
                    (folder_id, pid),
                ).fetchone()
                if row and row["kind"] == "user":
                    fl.move_file_to_folder(pid, meta.file_id, folder_id)

        item = {
            "filename": name,
            "status": "uploaded",
            "file_id": meta.file_id,
            "mime_type": meta.mime_type,
            "size_bytes": meta.size_bytes,
            "raw_hash": meta.raw_hash,
            "auto_kind": meta.auto_kind,
        }
        # Auto-promote .txt/.md/.markdown/.log into the data-prep source picker.
        from finetune_studio.data.fs.qa import maybe_auto_promote_upload
        source = maybe_auto_promote_upload(
            pid, meta.file_id, name, mime_type=meta.mime_type or ""
        )
        if source:
            item["source"] = source
            item["source_id"] = source.get("id")
        report.append(item)
        counts["uploaded"] += 1

    return JSONResponse({
        "ok": True,
        "counts": counts,
        "report": report,
    })


# ── File listing + metadata ──────────────────────────────────────────────

@router.get("/projects/{pid}/files")
async def list_files_route(
    pid: str,
    folder_id: str | None = None,
    include_deleted: bool = False,
    mime_prefix: str | None = None,
    search: str | None = None,
):
    _project_or_404(pid)
    files = fl.list_files(
        pid,
        folder_id=folder_id,
        include_deleted=include_deleted,
        mime_prefix=mime_prefix,
        search=search,
    )
    # Authoritative project-wide live count (ignores folder/search filters).
    # The "all files" tree badge must not use the filtered page length.
    total_count = len(fl.list_files(pid, include_deleted=False))
    return {"files": files, "count": len(files), "total_count": total_count}


# ── Trash endpoints (BEFORE /files/{fid} catch-all) ──────────────────────

@router.get("/projects/{pid}/files/trash")
async def list_trash_route(pid: str):
    _project_or_404(pid)
    return {"trash": fl.list_trash(pid)}


@router.post("/projects/{pid}/files/trash/purge")
async def purge_trash_route(pid: str, older_than_days: int = Query(7, ge=0)):
    _project_or_404(pid)
    return fl.purge_trash(pid, older_than_days=older_than_days)


# ── Pipeline status (BEFORE the /files/{fid} catch-all) ─────────────────

@router.get("/projects/{pid}/files/pipeline")
async def files_pipeline_route(pid: str):
    """Per-file workbench flags: {file_id: {has_parsed, source_id, chunk_count,
    parser, in_rag}} — drives the browser badges + editor affordances."""
    from finetune_studio.data.parsed_edit import pipeline_status
    _project_or_404(pid)
    return {"status": pipeline_status(pid)}


@router.post("/projects/{pid}/files/bulk")
async def files_bulk_route(pid: str, request: Request):
    """Bulk action over file ids: delete | restore | move | reparse | tag-add |
    tag-remove. Per-file isolation — one bad id never aborts the batch."""
    from finetune_studio.data.fs import workbench as wb
    _project_or_404(pid)
    body = await request.json()
    ids = [str(i) for i in (body.get("ids") or [])]
    action = str(body.get("action") or "")
    return wb.bulk_action(pid, ids, action, body)


@router.post("/projects/{pid}/files/download-zip")
async def files_download_zip_route(pid: str, request: Request):
    """Zip the raw bytes of the selected files (collision-safe entry names)."""
    from finetune_studio.data.fs import workbench as wb
    _project_or_404(pid)
    body = await request.json()
    ids = [str(i) for i in (body.get("ids") or [])]
    payload, filename = wb.download_zip(pid, ids)
    return Response(
        content=payload,
        media_type="application/zip",
        headers={"Content-Disposition": f'attachment; filename="{filename}"'},
    )


@router.get("/projects/{pid}/files/search-content")
async def files_search_content_route(pid: str, q: str = Query(""), limit: int = 25):
    """Substring search inside PARSED text of project files (name search
    stays client-side). Returns snippet per match."""
    from finetune_studio.data.fs import workbench as wb
    _project_or_404(pid)
    return wb.search_content(pid, q, limit=limit)


# ── Single-file routes (with {fid}) ──────────────────────────────────────

@router.get("/projects/{pid}/files/{fid}")
async def get_file_route(pid: str, fid: str):
    _project_or_404(pid)
    f = fl.get_file(pid, fid)
    if not f:
        raise HTTPException(status_code=404, detail="file not found")
    f["versions"] = fl.list_versions(pid, fid)
    f["conversions"] = fl.list_conversions(pid, fid)
    return f


@router.get("/projects/{pid}/files/{fid}/raw")
async def download_raw_route(
    pid: str,
    fid: str,
    version: int | None = None,
):
    """Download raw bytes. If version is None, serves the current version.
    Also works for files in trash (so the user can recover + inspect)."""
    from pathlib import Path
    _project_or_404(pid)
    f = fl.get_file(pid, fid, include_deleted=True)
    if not f:
        raise HTTPException(status_code=404, detail="file not found")
    versions = fl.list_versions(pid, fid)
    if not versions:
        raise HTTPException(status_code=404, detail="no versions")
    target_version = version if version is not None else f["current_version"]
    match = next((v for v in versions if v["version"] == target_version), None)
    if not match:
        raise HTTPException(
            status_code=404,
            detail=f"version {target_version} not found",
        )
    path = Path(match["raw_path"])
    if not path.exists():
        raise HTTPException(status_code=410, detail="file missing on disk")
    return FileResponse(
        path=str(path),
        filename=f["original_name"],
        media_type=f.get("mime_type") or "application/octet-stream",
    )


@router.get("/projects/{pid}/files/{fid}/parsed")
async def get_parsed_route(pid: str, fid: str):
    """Return the file's parsed-markdown representation.

    Response: ``{fid, path, parsed_md, source}`` where source is one of
    ``db`` | ``sibling`` | ``converted``. Binary formats without a stored
    conversion return HTTP 422.
    """
    _project_or_404(pid)
    return fl.get_parsed_markdown(pid, fid)


@router.put("/projects/{pid}/files/{fid}/parsed")
async def save_parsed_route(pid: str, fid: str, request: Request):
    """Save a hand-edited parsed text for a file (built-in editor).

    Writes the ``<raw>.md`` override and, when the file is a data-prep
    source, rewrites parsed.txt + regenerates chunks so training data and
    the next RAG build use the edited text. Raw bytes stay immutable.
    """
    from finetune_studio.data.parsed_edit import save_parsed_override
    _project_or_404(pid)
    body = await request.json()
    text = body.get("text")
    if text is None:
        raise HTTPException(status_code=400, detail="text required")
    return save_parsed_override(pid, fid, str(text))


@router.post("/projects/{pid}/files/{fid}/reparse")
async def reparse_file_route(pid: str, fid: str):
    """Discard the manual parsed override and re-run the real parser."""
    from finetune_studio.data.parsed_edit import reparse_file
    _project_or_404(pid)
    return reparse_file(pid, fid)


@router.get("/projects/{pid}/files/{fid}/usage")
async def file_usage_route(pid: str, fid: str):
    """Where this file's content actually went: prep source, RAG corpus,
    QA pairs, datasets built from them, training runs that used those datasets."""
    from finetune_studio.data.fs import workbench as wb
    _project_or_404(pid)
    return wb.file_usage(pid, fid)


@router.get("/projects/{pid}/files/{fid}/versions")
async def list_versions_route(pid: str, fid: str):
    _project_or_404(pid)
    return {"versions": fl.list_versions(pid, fid)}


@router.get("/projects/{pid}/files/{fid}/conversions")
async def list_conversions_route(pid: str, fid: str):
    _project_or_404(pid)
    return {"conversions": fl.list_conversions(pid, fid)}


@router.patch("/projects/{pid}/files/{fid}/rename")
async def rename_file_route(pid: str, fid: str, request: Request):
    """Rename a live file (DB original_name + on-disk path).

    Body accepts ``new_name`` (preferred) or ``name`` (data-prep UI compat).
    """
    _project_or_404(pid)
    body = await request.json()
    new_name = body.get("new_name")
    if new_name is None:
        new_name = body.get("name")
    if new_name is None:
        raise HTTPException(status_code=400, detail="new_name required")
    return fl.rename_file(pid, fid, str(new_name))


@router.post("/projects/{pid}/files/{fid}/purge")
async def purge_file_route(pid: str, fid: str):
    """Hard-delete one trashed file (disk + DB). Must already be in trash."""
    _project_or_404(pid)
    return fl.purge_file(pid, fid)


@router.post("/projects/{pid}/files/{fid}/move")
async def move_file_route(pid: str, fid: str, request: Request):
    _project_or_404(pid)
    body = await request.json()
    folder_id = body.get("folder_id")
    if not folder_id:
        raise HTTPException(status_code=400, detail="folder_id required")
    return fl.move_file_to_folder(pid, fid, folder_id)


@router.delete("/projects/{pid}/files/{fid}")
async def delete_file_route(pid: str, fid: str):
    _project_or_404(pid)
    return fl.soft_delete_file(pid, fid)


@router.post("/projects/{pid}/files/{fid}/restore")
async def restore_file_route(pid: str, fid: str):
    _project_or_404(pid)
    return fl.restore_file(pid, fid)


# ── Folder CRUD ──────────────────────────────────────────────────────────

@router.post("/projects/{pid}/folders")
async def create_folder_route(pid: str, request: Request):
    _project_or_404(pid)
    body = await request.json()
    name = (body.get("name") or "").strip()
    if not name:
        raise HTTPException(status_code=400, detail="name required")
    return fl.create_folder(pid, name)


@router.get("/projects/{pid}/folders")
async def list_folders_route(pid: str, include_auto: bool = Query(True)):
    _project_or_404(pid)
    return {"folders": fl.list_folders(pid, include_auto=include_auto)}


@router.patch("/projects/{pid}/files/{fid}/tags")
async def update_file_tags(pid: str, fid: str, request: Request):
    """Update tags and notes for a file."""
    from finetune_studio import db
    _project_or_404(pid)
    body = await request.json()
    tags = body.get("tags", "")
    notes = body.get("notes", "")
    with db.cursor() as c:
        c.execute(
            "UPDATE project_files SET tags = ?, notes = ? "
            "WHERE id = ? AND project_id = ?",
            (tags, notes, fid, pid),
        )
    return {"ok": True, "tags": tags, "notes": notes}


@router.patch("/projects/{pid}/folders/{fid}")
async def rename_folder_route(pid: str, fid: str, request: Request):
    _project_or_404(pid)
    body = await request.json()
    name = (body.get("name") or "").strip()
    if not name:
        raise HTTPException(status_code=400, detail="name required")
    return fl.rename_folder(pid, fid, name)


@router.delete("/projects/{pid}/folders/{fid}")
async def delete_folder_route(pid: str, fid: str):
    _project_or_404(pid)
    return fl.delete_folder(pid, fid)
