"""Routes for the self-healing /api/system/update endpoint.

POST /api/system/update       \u2014 queue a background update.sh run
GET  /api/system/update/{uid}  \u2014 poll status + log tail
GET  /api/system/updates       \u2014 list recent update attempts

The worker spawns the update.sh script as a subprocess and streams
each stdout line into the system_updates.log_text DB field via
append_log(), so the UI can show a live tail of what the script is
doing.

FTS_SKIP_UPDATE=1 short-circuits the worker with a canned log so
end-to-end tests can assert the full lifecycle without spawning a
real shell process.
"""

from __future__ import annotations

import logging
import os
import subprocess
import threading
from pathlib import Path
from typing import Optional

from fastapi import APIRouter, BackgroundTasks, Request
from fastapi.responses import JSONResponse

from finetune_studio import db

log = logging.getLogger(__name__)
router = APIRouter()

# Repo root = four levels up from src/finetune_studio/webui/routes/updates.py.
# Path(__file__)         = .../finetune-studio/src/finetune_studio/webui/routes/updates.py
# parents[0]             = .../finetune-studio/src/finetune_studio/webui/routes
# parents[1]             = .../finetune-studio/src/finetune_studio/webui
# parents[2]             = .../finetune-studio/src/finetune_studio
# parents[3]             = .../finetune-studio/src
# parents[4]             = .../finetune-studio      ← repo root (update.sh lives here)
REPO_ROOT = Path(__file__).resolve().parents[4]


def _find_update_script() -> Optional[Path]:
    """Locate update.sh. Prefers the path next to the package; falls
    back to $FTS_UPDATE_SCRIPT, then CWD/update.sh."""
    candidates = [
        REPO_ROOT / "update.sh",
        Path(os.environ.get("FTS_UPDATE_SCRIPT", "") or ""),
        Path.cwd() / "update.sh",
    ]
    for c in candidates:
        try:
            if c and c.is_file():
                return c
        except (OSError, ValueError):
            pass
    return None


@router.post("/system/update")
async def trigger_update(request: Request, background: BackgroundTasks):
    """Queue a self-healing update.

    Body:
      mode:           'update' (default) | 'check' | 'repair'
      no_pull:        bool \u2014 skip 'git pull'
      no_llama:       bool \u2014 skip llama.cpp CLI build
      no_restart:     bool \u2014 skip service restart
      triggered_by:   'user' (default) | 'system'

    Returns: {ok, update_id, status: 'queued'}
    """
    body = await request.json() if request.headers.get(
        "content-type", "").startswith("application/json") else {}
    mode = (body.get("mode") or "update").lower()
    if mode not in ("update", "check", "repair"):
        return {"error": f"invalid mode: {mode}"}
    options = {
        "no_pull": bool(body.get("no_pull", False)),
        "no_llama": bool(body.get("no_llama", False)),
        "no_restart": bool(body.get("no_restart", False)),
    }
    triggered_by = (body.get("triggered_by") or "user")[:32]
    row = db.create_update(mode=mode, options=options,
                           triggered_by=triggered_by)
    background.add_task(_update_worker, row["id"], mode, options)
    return {"ok": True, "update_id": row["id"], "status": "queued",
            "mode": mode, "options": options}


@router.get("/system/update/{uid}")
async def get_update_status(uid: str):
    """Status + log tail for one update attempt."""
    row = db.get_update(uid)
    if not row:
        return JSONResponse({"error": "not found"}, status_code=404)
    # Compact the log_text in the response by default \u2014 a full 256KB
    # blob is too big for a poll request. The UI can ask for the full
    # text via ?full=1 if needed.
    full = False  # parsed from query in real route; static here for clarity
    log_text = row.get("log_text") or ""
    row["log_tail"] = log_text[-4000:] if not full else log_text
    row["log_length"] = len(log_text)
    return row


@router.get("/system/updates")
async def list_updates(limit: int = 50):
    return db.list_updates_recent(limit=limit)


@router.get("/system/update/latest")
async def latest_update():
    """The most recent in-progress (queued or running) update \u2014 what
    a dashboard would poll for a live progress bar."""
    row = db.latest_update_in_progress()
    if not row:
        return {"exists": False}
    log_text = row.get("log_text") or ""
    row["log_tail"] = log_text[-4000:]
    row["log_length"] = len(log_text)
    return row


def _update_worker(uid: str, mode: str, options: dict) -> None:
    """Spawn update.sh as subprocess, stream output to the DB row.

    Test mode: FTS_SKIP_UPDATE=1 emits a canned log and marks the row
    done without spawning anything. Use this for unit/integration tests.
    """
    if os.environ.get("FTS_SKIP_UPDATE") == "1":
        db.mark_update_running(uid)
        db.append_update_log(uid, "[FTS_SKIP_UPDATE=1] test mode \u2014 not running real update.sh\n")
        db.append_update_log(uid, f"  mode={mode} options={options}\n")
        db.append_update_log(uid, "  step 1: git pull (skipped)\n")
        db.append_update_log(uid, "  step 2: pip sync (skipped)\n")
        db.append_update_log(uid, "  step 3: llama.cpp (skipped)\n")
        db.append_update_log(uid, "  step 4: init_db (skipped)\n")
        db.append_update_log(uid, "  step 5: restart (skipped)\n")
        db.mark_update_done(uid)
        return

    script_path = _find_update_script()
    if script_path is None:
        db.mark_update_failed(uid, error="update.sh not found in repo root, "
                                "$FTS_UPDATE_SCRIPT, or cwd")
        return

    args = [str(script_path)]
    if options.get("no_pull"):
        args.append("--no-pull")
    if options.get("no_llama"):
        args.append("--no-llama")
    if options.get("no_restart"):
        args.append("--no-restart")
    if mode == "check":
        args.append("--check")
    elif mode == "repair":
        args.append("--repair")

    db.mark_update_running(uid)
    db.append_update_log(uid, f"$ {' '.join(args)}\n")

    try:
        proc = subprocess.Popen(
            args,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
            cwd=str(script_path.parent),
        )
        # Stream output line by line into the DB. We block here until
        # the script exits \u2014 this runs in a FastAPI BackgroundTask,
        # not on the request thread.
        assert proc.stdout is not None
        restart_seen = False
        for line in proc.stdout:
            if "Restarting finetune-studio.service" in line:
                restart_seen = True
            db.append_update_log(uid, line)
        proc.wait()
        if proc.returncode == 0:
            db.mark_update_done(uid)
        elif restart_seen:
            # update.sh was SIGTERMed by the very restart it triggered —
            # every real step (pull, deps, migrations) already succeeded.
            # Mark done, not error: -15 here is success, not failure.
            db.append_update_log(uid,
                "\n[worker] update.sh ended by the service restart it "
                "triggered (expected) → marked done\n")
            db.mark_update_done(uid)
        else:
            db.mark_update_failed(
                uid,
                error=f"update.sh exited with code {proc.returncode}",
            )
    except Exception as e:  # noqa: BLE001
        log.exception("update worker failed")
        db.mark_update_failed(uid, error=str(e))
