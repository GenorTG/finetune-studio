"""Project settings API — WebUI log tail for the settings page.

Reads the bare-uvicorn stdout/stderr log so Genor can inspect WebUI
output without SSH. Primary path is ``/tmp/uvicorn.log`` (fan-dragon);
falls back to home and systemd log locations.
"""
from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from fastapi import APIRouter, HTTPException, Query

router = APIRouter(tags=["project-settings"])

LOG_CANDIDATES: tuple[str, ...] = (
    "/tmp/uvicorn.log",
    "/home/genortg/.finetune-studio/uvicorn.log",
    "/var/log/finetune-studio/uvicorn.log",
)

DEFAULT_LINES = 80
MAX_LINES = 500


def resolve_log_path(candidates: tuple[str, ...] | None = None) -> str:
    """Return the first existing candidate path, else the primary candidate."""
    paths = candidates if candidates is not None else LOG_CANDIDATES
    for p in paths:
        if Path(p).is_file():
            return p
    return paths[0] if paths else "/tmp/uvicorn.log"


def read_log_tail(path: str, n: int) -> list[str]:
    """Return the last ``n`` lines of ``path``.

    If the file is missing or unreadable, returns a one-element fallback
    message list so the UI always has something to show.
    """
    p = Path(path)
    if not p.is_file():
        return [f"(no log file found at {path})"]
    try:
        text = p.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return [f"(no log file found at {path})"]
    if n <= 0:
        return []
    return text.splitlines()[-n:]


def _iso_now() -> str:
    """UTC timestamp for the response ``updated_at`` field."""
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def build_logs_payload(
    lines: int,
    *,
    candidates: tuple[str, ...] | None = None,
) -> dict[str, Any]:
    """Build the JSON body for ``GET .../logs``."""
    n = max(1, min(int(lines), MAX_LINES))
    path = resolve_log_path(candidates)
    return {
        "lines": read_log_tail(path, n),
        "path": path,
        "updated_at": _iso_now(),
    }


@router.get("/projects/{pid}/logs")
async def project_logs(
    pid: str,
    lines: int = Query(DEFAULT_LINES, ge=1, le=MAX_LINES),
) -> dict[str, Any]:
    """Return the last ``lines`` of the WebUI uvicorn log (default 80, max 500)."""
    from finetune_studio import db

    if not db.get_project(pid):
        raise HTTPException(status_code=404, detail="project not found")
    return build_logs_payload(lines)
