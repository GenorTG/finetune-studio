"""Data tab — list, upload, validate, preview, deduplicate flat data files."""

import os

import aiofiles
from fastapi import APIRouter, File, UploadFile
from fastapi.responses import JSONResponse

from finetune_studio.config import settings
from finetune_studio.data.organizer import dedup_data, scan_data_files
from finetune_studio.data.validator import validate_file
from finetune_studio.training.data import load_jsonl

router = APIRouter()

@router.get("/files")
async def list_files():
    return scan_data_files(settings.data_dir)

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
    return validate_file(path)

@router.get("/preview")
async def preview(path: str, limit: int = 10):
    try:
        data = load_jsonl(path)
        return {"rows": len(data), "preview": data[: max(0, limit)]}
    except Exception as e:  # noqa: BLE001
        return {"error": str(e)}

@router.post("/dedup")
async def dedup(path: str):
    try:
        data = load_jsonl(path)
    except Exception as e:  # noqa: BLE001
        return JSONResponse({"error": str(e)}, status_code=400)
    unique, dupes = dedup_data(data)
    return {"original": len(data), "unique": len(unique), "removed": dupes}
