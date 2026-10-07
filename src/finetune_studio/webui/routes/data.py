"""Flat data files — upload, validate, deduplicate."""

import asyncio
import os
from pathlib import Path

import aiofiles
from fastapi import APIRouter, File, UploadFile
from fastapi.responses import JSONResponse

from finetune_studio.config import settings
from finetune_studio.data.fs.paths import resolve_within
from finetune_studio.data.organizer import dedup_data
from finetune_studio.data.validator import validate_file
from finetune_studio.training.data import load_jsonl

router = APIRouter()


def _data_path(path: str) -> str:
    """Confine a client-supplied path to ``settings.data_dir`` (400/403 otherwise)."""
    return str(resolve_within(path, [Path(settings.data_dir)], what="path"))


@router.post("/upload")
async def upload_file(file: UploadFile = File(...)):  # noqa: B008
    # Strip any directory components from the client-supplied filename —
    # otherwise a name like "../../etc/cron.d/x" escapes settings.data_dir.
    safe_name = os.path.basename(file.filename or "")
    if safe_name in ("", ".", ".."):
        safe_name = "upload"
    dest = os.path.join(settings.data_dir, safe_name)
    content = await file.read()
    async with aiofiles.open(dest, "wb") as f:
        await f.write(content)
    return {"path": dest, "name": safe_name, "size": len(content)}

@router.get("/validate")
async def validate(path: str):
    return await asyncio.to_thread(validate_file, _data_path(path))

@router.post("/dedup")
async def dedup(path: str):
    path = _data_path(path)
    try:
        data = await asyncio.to_thread(load_jsonl, path)
    except Exception as e:  # noqa: BLE001
        return JSONResponse({"error": str(e)}, status_code=400)
    unique, dupes = await asyncio.to_thread(dedup_data, data)
    return {"original": len(data), "unique": len(unique), "removed": dupes}
