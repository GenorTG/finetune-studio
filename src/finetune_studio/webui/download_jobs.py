"""Hugging Face model downloads as supervised job processes: a web restart no longer ends them.

``start`` creates the ``hf_downloads`` row (status ``queued``) and launches job ``download-<download id>``
(``jobs.kinds.download``, no GPU). The job process runs the hub call on a thread and writes the row: ``downloading``
with live byte counts (what is on disk under the destination, against the size the Hub reports), then
``completed`` / ``error``. The row is the one truth the HF page, the activity drawer and a restarted web app read.

Cancel: the web side writes ``cancelled`` (only while the row is still active, one SQL statement) and stops the
job (SIGTERM); the job then exits without waiting for the hub call. Partial files stay in the destination's
``.cache/huggingface`` folder, so the next download of the same file resumes them.
"""
from __future__ import annotations

import logging
import sys
import threading
from pathlib import Path
from typing import Any

from finetune_studio.db import hf_downloads as dl_db
from finetune_studio.jobs import launch, store

log = logging.getLogger(__name__)

KIND = "download"
STOP_GRACE_S = 10.0
TERMINAL_STATUSES = frozenset({"completed", "error", "cancelled"})
PROGRESS_EVERY_S = 1.0
RESTART_ERROR = "service restarted before download finished"


def job_id(download_id: str) -> str:
    return f"{KIND}-{download_id}"


# ── web side ────────────────────────────────────────────────────────────

def start(*, repo_id: str, filename: str | None, revision: str, dest: str) -> dict[str, Any]:
    """Create the row and launch its job; returns the queued row. Raises ``launch.JobRejected``."""
    row = dl_db.create_job(repo_id=repo_id, filename=filename or "")
    did = row["id"]
    spec = store.JobSpec(
        id=job_id(did), kind=KIND, gpu=False, stop_grace_s=STOP_GRACE_S,
        title=f"download {repo_id}" + (f" / {filename}" if filename else ""),
        params={"download_id": did, "repo_id": repo_id, "filename": filename or "", "revision": revision,
                "dest": dest},
    )
    try:
        launch.start(spec)
    except launch.JobRejected as e:
        dl_db.finish_if_active(did, "error", error=f"not started: {e}")
        raise
    return row


def _job_alive(download_id: str) -> bool | None:
    """None = this row has no job (a download from before downloads were jobs)."""
    try:
        return bool(launch.describe(job_id(download_id))["running"])
    except (FileNotFoundError, ValueError):
        return None


def _job_error(download_id: str) -> str:
    try:
        ex = launch.describe(job_id(download_id)).get("exit") or {}
    except (FileNotFoundError, ValueError):
        return ""
    return str(ex.get("error") or ex.get("detail") or "")


def _settle(row: dict[str, Any]) -> dict[str, Any]:
    """A row still active whose job has ended without finishing it (crashed, killed): record that, honestly."""
    if row.get("status") in TERMINAL_STATUSES or _job_alive(row["id"]) is not False:
        return row
    reason = _job_error(row["id"]) or "the download process ended without a result"
    if dl_db.finish_if_active(row["id"], "error", error=f"download job ended: {reason}"):
        log.warning("download %s: job ended without finishing the row (%s)", row["id"], reason)
    return dl_db.get_job(row["id"]) or row


def snapshot(row: dict[str, Any]) -> dict[str, Any]:
    """The progress shape the pages read (``GET /hf/download/progress``, the activity drawer)."""
    return {
        "job_id": row["id"],
        "status": row.get("status") or "queued",
        "repo_id": row.get("repo_id") or "",
        "filename": row.get("filename") or "",
        "created_at": row.get("created_at"),
        "started_at": row.get("started_at"),
        "finished_at": row.get("finished_at"),
        "bytes_done": row.get("bytes_done") or 0,
        "bytes_total": row.get("bytes_total"),
        "path": row.get("path") or None,
        "error": row.get("error") or None,
        "running": bool(_job_alive(row["id"])),
    }


def progress(download_id: str) -> dict[str, Any] | None:
    row = dl_db.get_job(download_id)
    return None if row is None else snapshot(_settle(row))


def cancel(download_id: str) -> str | None:
    """Cancel a download; returns its status afterwards (None = unknown id). A finished one is left as it is."""
    row = dl_db.get_job(download_id)
    if row is None:
        return None
    if row.get("status") in TERMINAL_STATUSES:
        return str(row["status"])
    dl_db.finish_if_active(download_id, "cancelled", error="user cancelled")
    if _job_alive(download_id):
        launch.stop(job_id(download_id))     # SIGTERM -> the job leaves the hub call behind and exits
    return str((dl_db.get_job(download_id) or {}).get("status") or "cancelled")


def running_download_ids() -> set[str]:
    """Rows whose job process is alive (a web restart left them running)."""
    out = set()
    for j in launch.list_jobs(KIND, running_only=True):
        try:
            out.add(str(store.read_spec(j["id"]).params["download_id"]))
        except (FileNotFoundError, ValueError, KeyError):
            continue
    return out


def live_downloads() -> list[dict[str, Any]]:
    """In-flight downloads whose job is running, for the activity drawer."""
    keep = running_download_ids()
    return [snapshot(r) for r in dl_db.list_in_progress(limit=200) if r["id"] in keep]


def reconcile_on_startup() -> int:
    """Web startup: a queued/downloading row whose job is gone will never progress -> error. Rows whose job is
    alive (the web restarted, the download did not) are left alone. Returns the number of rows failed."""
    keep = running_download_ids()
    failed = 0
    for row in dl_db.list_in_progress(limit=200):
        if row["id"] in keep:
            continue
        reason = _job_error(row["id"])
        if dl_db.finish_if_active(row["id"], "error", error=RESTART_ERROR + (f" ({reason})" if reason else "")):
            failed += 1
    if failed:
        log.info("Marked %d interrupted HF download(s) as failed", failed)
    return failed


def on_finished(job: dict[str, Any]) -> None:
    """Web process (``jobs.watch``): a download job ended; settle its row and show the new model in the pickers."""
    try:
        did = str(store.read_spec(job["id"]).params["download_id"])
    except (FileNotFoundError, ValueError, KeyError):
        return
    row = progress(did)
    if row and row["status"] == "completed":
        _refresh_registry()


def _refresh_registry() -> None:
    """Web process only: a job process has no registry (and must not import the web app)."""
    if "finetune_studio.webui.app" not in sys.modules:
        return
    try:
        from finetune_studio.webui.routes.models import refresh_model_registry
        n = refresh_model_registry()
        log.info("HF download complete — registry now has %d model(s)", n)
    except Exception:
        log.exception("registry refresh after HF download failed")


# ── job side ────────────────────────────────────────────────────────────

def _hub_fetch(repo_id: str, filename: str, revision: str, dest: str) -> str:
    """One file (``filename``) or the whole repo into ``dest``; returns the local path."""
    from huggingface_hub import hf_hub_download, snapshot_download
    if filename:
        return hf_hub_download(repo_id=repo_id, filename=filename, revision=revision, local_dir=dest)
    return snapshot_download(repo_id=repo_id, revision=revision, local_dir=dest)


def _remote_size(repo_id: str, filename: str, revision: str) -> int | None:
    """Bytes the Hub reports for the file / every file of the repo; None when it does not say."""
    try:
        from huggingface_hub import HfApi, get_hf_file_metadata, hf_hub_url
        if filename:
            size = get_hf_file_metadata(hf_hub_url(repo_id, filename, revision=revision)).size
            return int(size) if size else None
        info = HfApi().model_info(repo_id, revision=revision, files_metadata=True)
        return sum(int(s.size or 0) for s in (info.siblings or [])) or None
    except Exception:                  # progress is a nicety; the download itself reports real failures
        log.debug("could not read the remote size of %s", repo_id, exc_info=True)
        return None


def _bytes_on_disk(dest: Path) -> int:
    """Everything under ``dest``, the hub's ``.incomplete`` partials included (files may vanish mid-walk)."""
    total = 0
    for p in dest.rglob("*"):
        try:
            if p.is_file():
                total += p.stat().st_size
        except OSError:
            continue
    return total


def run_in_job(params: dict[str, Any], stop: threading.Event) -> str:
    """The job process side; returns done | cancelled | failed. ``stop`` is the job's stop event (SIGTERM)."""
    did = str(params["download_id"])
    repo_id, filename, revision = str(params["repo_id"]), str(params.get("filename") or ""), str(params["revision"])
    dest = Path(params["dest"])
    if not dl_db.start_if_queued(did):
        return "cancelled"                    # cancelled before the job got going
    dest.mkdir(parents=True, exist_ok=True)
    total = _remote_size(repo_id, filename, revision)
    outcome: dict[str, Any] = {}

    def fetch() -> None:
        try:
            outcome["path"] = _hub_fetch(repo_id, filename, revision, str(dest))
        except BaseException as e:  # noqa: BLE001 - reported as the row's error below
            outcome["error"] = e

    worker = threading.Thread(target=fetch, daemon=True, name=f"hf-download-{did}")
    worker.start()
    while worker.is_alive():
        dl_db.set_progress(did, bytes_done=min(_bytes_on_disk(dest), total or 1 << 62), bytes_total=total)
        if stop.wait(PROGRESS_EVERY_S):
            break
    if worker.is_alive():                     # stopped: the hub call is left behind, the process exits
        dl_db.finish_if_active(did, "cancelled", error="user cancelled")
        return "cancelled"
    if "error" in outcome:
        e = outcome["error"]
        log.error("download %s of %s failed: %s", did, repo_id, e)
        dl_db.finish_if_active(did, "error", error=str(e) or type(e).__name__)
        return "failed"
    path = Path(outcome["path"])
    size = path.stat().st_size if path.is_file() else _bytes_on_disk(path)
    if not dl_db.finish_if_active(did, "completed", path=str(path), bytes_total=size, bytes_done=size):
        return "cancelled"                    # the user cancelled while the last bytes arrived
    _refresh_registry()                       # in-process jobs (tests); a job process leaves it to on_finished
    return "done"
