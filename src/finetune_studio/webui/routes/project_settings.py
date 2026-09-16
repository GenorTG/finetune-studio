"""Project settings API — WebUI log tail for the settings page.

Prefer the live ``finetune-studio`` systemd user journal when that unit is
active (fan-dragon production). Bare ``/tmp/uvicorn.log`` from an old
non-service process is treated as stale/unavailable so the UI never claims
"live" for the wrong PID.
"""
from __future__ import annotations

import logging
import subprocess
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from fastapi import APIRouter, HTTPException, Query

log = logging.getLogger(__name__)
router = APIRouter(tags=["project-settings"])

LOG_CANDIDATES: tuple[str, ...] = (
    "/tmp/uvicorn.log",
    "/home/genortg/.finetune-studio/uvicorn.log",
    "/var/log/finetune-studio/uvicorn.log",
)

SYSTEMD_UNIT = "finetune-studio.service"
# File mtime older than this (seconds) is treated as stale when systemd owns the app.
STALE_FILE_SECONDS = 120

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
    return datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")


def systemd_unit_active(unit: str = SYSTEMD_UNIT) -> bool:
    """True when the user systemd unit is reported active."""
    try:
        proc = subprocess.run(
            ["systemctl", "--user", "is-active", unit],
            capture_output=True,
            text=True,
            timeout=2,
            check=False,
        )
    except (OSError, subprocess.TimeoutExpired):
        return False
    return proc.returncode == 0 and (proc.stdout or "").strip() == "active"


def journal_tail(n: int, *, unit: str = SYSTEMD_UNIT) -> list[str] | None:
    """Return last ``n`` journal lines for the unit, or None if unavailable."""
    try:
        proc = subprocess.run(
            [
                "journalctl",
                "--user",
                "-u",
                unit,
                "-n",
                str(n),
                "--no-pager",
                "-o",
                "cat",
            ],
            capture_output=True,
            text=True,
            timeout=5,
            check=False,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        log.debug("journalctl unavailable: %s", exc)
        return None
    if proc.returncode != 0:
        return None
    lines = (proc.stdout or "").splitlines()
    if lines:
        return lines
    return [f"(systemd unit {unit} active but journal has no lines yet)"]


def file_mtime_age_seconds(path: str) -> float | None:
    """Seconds since mtime, or None if missing/unreadable."""
    try:
        return max(0.0, datetime.now(UTC).timestamp() - Path(path).stat().st_mtime)
    except OSError:
        return None


def build_logs_payload(
    lines: int,
    *,
    candidates: tuple[str, ...] | None = None,
    prefer_systemd: bool | None = None,
) -> dict[str, Any]:
    """Build the JSON body for ``GET .../logs``.

    Contract:
    - ``live`` true only for a source believed to be the running WebUI.
    - When the systemd unit is active, prefer ``journalctl --user -u …``.
    - A bare ``/tmp/uvicorn.log`` that is old while systemd is active is
      reported as ``stale`` / not live (never claimed as the live service log).
    """
    n = max(1, min(int(lines), MAX_LINES))
    use_systemd = systemd_unit_active() if prefer_systemd is None else prefer_systemd

    if use_systemd:
        journal_lines = journal_tail(n)
        if journal_lines is not None:
            return {
                "lines": journal_lines,
                "path": f"journalctl --user -u {SYSTEMD_UNIT}",
                "source": "systemd",
                "live": True,
                "stale": False,
                "updated_at": _iso_now(),
            }
        # Unit active but journal unreadable — do not silently serve a stale file.
        path = resolve_log_path(candidates)
        age = file_mtime_age_seconds(path)
        stale = age is None or age > STALE_FILE_SECONDS
        if stale:
            msg = (
                f"(systemd unit {SYSTEMD_UNIT} is active, but journalctl is "
                f"unavailable and {path} looks stale/unavailable — not claiming live)"
            )
            return {
                "lines": [msg],
                "path": path,
                "source": "unavailable",
                "live": False,
                "stale": True,
                "updated_at": _iso_now(),
            }

    path = resolve_log_path(candidates)
    age = file_mtime_age_seconds(path)
    exists = Path(path).is_file()
    if not exists:
        return {
            "lines": [f"(no log file found at {path})"],
            "path": path,
            "source": "file",
            "live": False,
            "stale": False,
            "updated_at": _iso_now(),
        }

    # Dev / non-systemd: treat a recently written file as live.
    live = age is not None and age <= STALE_FILE_SECONDS
    stale = not live
    body = read_log_tail(path, n)
    if stale:
        notice = (
            f"(log file may be stale — mtime age {int(age or 0)}s; "
            f"prefer journalctl --user -u {SYSTEMD_UNIT} when the service owns :7860)"
        )
        body = [notice, *body]
    return {
        "lines": body,
        "path": path,
        "source": "file",
        "live": live,
        "stale": stale,
        "updated_at": _iso_now(),
    }


@router.get("/projects/{pid}/logs")
async def project_logs(
    pid: str,
    lines: int = Query(DEFAULT_LINES, ge=1, le=MAX_LINES),
) -> dict[str, Any]:
    """Return the last ``lines`` of the WebUI process log (default 80, max 500)."""
    from finetune_studio import db

    if not db.get_project(pid):
        raise HTTPException(status_code=404, detail="project not found")
    return build_logs_payload(lines)
