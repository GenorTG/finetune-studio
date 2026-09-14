"""Project models API — export dir contents for the Models expand-row."""
from __future__ import annotations

import os
from pathlib import Path

from fastapi import APIRouter, HTTPException

router = APIRouter(tags=["project-models"])

_MAX_CONTENTS = 20


def list_dir_contents(path: str, *, limit: int = _MAX_CONTENTS) -> list[dict]:
    """Return up to ``limit`` top-level entries in ``path`` with sizes.

    Each item: ``{"name": str, "size_bytes": int, "is_dir": bool}``.
    Directories report size 0 (no recursive walk). Sorted by name.
    """
    entries: list[dict] = []
    try:
        names = sorted(os.listdir(path))
    except OSError:
        return entries
    for name in names:
        if len(entries) >= limit:
            break
        full = os.path.join(path, name)
        try:
            is_dir = os.path.isdir(full)
            size = 0 if is_dir else os.path.getsize(full)
        except OSError:
            continue
        entries.append({"name": name, "size_bytes": size, "is_dir": is_dir})
    return entries


def _normalize_export_path(raw: str) -> str:
    """Turn a path:path capture into an absolute filesystem path."""
    p = (raw or "").strip()
    if not p:
        return ""
    # URL path capture drops the leading slash for absolute paths.
    if not p.startswith("/") and not (len(p) >= 2 and p[1] == ":"):
        p = "/" + p
    return os.path.normpath(p)


def export_belongs_to_project(pid: str, export_path: str) -> bool:
    """True if ``export_path`` sits under a training run of ``pid``."""
    from finetune_studio import db

    if not export_path:
        return False
    try:
        target = Path(export_path).resolve()
    except OSError:
        return False
    for run in db.list_runs(pid):
        out = (run.get("output_path") or "").strip()
        if not out:
            continue
        try:
            root = Path(out).resolve()
        except OSError:
            continue
        if target == root or root in target.parents:
            return True
    return False


@router.get("/projects/{pid}/models/{export_path:path}/contents")
async def model_export_contents(pid: str, export_path: str) -> dict:
    """List top-level files in a trained export directory (max 20)."""
    from finetune_studio import db

    if not db.get_project(pid):
        raise HTTPException(status_code=404, detail="project not found")

    path = _normalize_export_path(export_path)
    if not path or not os.path.isdir(path):
        raise HTTPException(status_code=404, detail="export path not found")
    if not export_belongs_to_project(pid, path):
        raise HTTPException(status_code=403, detail="path not in project")

    files = list_dir_contents(path)
    return {"path": path, "files": files, "truncated": len(files) >= _MAX_CONTENTS}
