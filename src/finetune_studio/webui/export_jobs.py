"""Lifecycle of WebUI export jobs: one worker thread, one row, one at a time.

An export (merge, GGUF convert, quantize, abliterate) takes minutes and
saturates the GPU or CPU, so it never runs on the event loop and never inside
an HTTP request. ``start_job`` creates the ``model_exports`` row (status
``queued``) and hands the work to a daemon thread; the row then carries the
real state — ``status``, ``phase`` + ``phase_detail`` (reported by the
pipeline through :mod:`finetune_studio.training.export_job`), a
``heartbeat_at`` timestamp and the error text — so a page reload, a second
tab or the SSE stream all read the same truth.

Concurrency policy: **single flight, 409 on conflict.** Every export shares
the one GPU and one llama.cpp toolchain; a second export while one is queued or
running is refused with :class:`ExportBusy` (HTTP 409 naming the active job)
instead of silently queueing behind a multi-minute job. The Export page
disables its buttons while a job is active and runs several selected formats
one after the other.

Cancel stops a job at its next checkpoint: immediately while a llama.cpp child
runs (its whole process group is killed), but only between phases while the
in-process adapter merge is running (a torch merge cannot be interrupted).
"""

from __future__ import annotations

import atexit
import logging
import threading
from collections.abc import Callable
from dataclasses import dataclass

from finetune_studio import db
from finetune_studio.training import export_job

log = logging.getLogger(__name__)

TERMINAL_STATUSES = frozenset({"done", "failed", "error", "cancelled"})

# ``work`` runs inside the bound ExportContext on the worker thread. It does
# the export, finishes every row it owns with ``db.mark_export_done`` and
# raises on failure (the message becomes the row's ``error``).
Work = Callable[[export_job.ExportContext], None]


class ExportBusy(RuntimeError):
    """Another export is queued or running (single-flight policy)."""

    def __init__(self, active: dict) -> None:
        super().__init__(
            f"another export is already {active.get('status') or 'running'} "
            f"(id {active.get('id')})"
        )
        self.active = active


@dataclass
class _Job:
    export_id: str
    ctx: export_job.ExportContext
    thread: threading.Thread | None = None


_LOCK = threading.Lock()
_ACTIVE: dict[str, _Job] = {}


def active_ids() -> list[str]:
    with _LOCK:
        return list(_ACTIVE)


def is_active(export_id: str) -> bool:
    with _LOCK:
        return export_id in _ACTIVE


def start_job(
    *,
    project_id: str,
    run_id: str,
    fmt: str,
    quant: str,
    quants: list[str] | None,
    make_work: Callable[[str], Work],
) -> dict:
    """Create the export row and start its worker; return the queued row.

    ``make_work(export_id)`` builds the work callable once the id is known.
    Raises :class:`ExportBusy` when a job is already in flight.
    """
    with _LOCK:
        if _ACTIVE:
            eid = next(iter(_ACTIVE))
            raise ExportBusy(db.get_export(eid) or {"id": eid, "status": "running"})
        row = db.create_export(
            project_id=project_id, run_id=run_id, format=fmt, quant=quant,
            quants=quants,
        )
        eid = row["id"]
        ctx = export_job.ExportContext(
            eid,
            on_phase=lambda phase, detail: db.set_export_phase(eid, phase, detail),
            on_heartbeat=lambda: db.heartbeat_export(eid),
        )
        job = _Job(eid, ctx)
        job.thread = threading.Thread(
            target=_run, args=(job, make_work(eid)), daemon=True,
            name=f"export-{eid[:8]}",
        )
        _ACTIVE[eid] = job
        job.thread.start()
    return row


def cancel_job(export_id: str) -> bool:
    """Ask a running job to stop; False when it is not active in this process."""
    with _LOCK:
        job = _ACTIVE.get(export_id)
    if job is None:
        return False
    # The runner polls this flag every 0.25 s and kills the child's group
    # itself; no blocking kill here (called from an async handler).
    job.ctx.cancel_event.set()
    try:
        db.set_export_phase(export_id, "cancelling", "stopping…")
    except Exception:
        log.debug("could not record cancelling phase", exc_info=True)
    return True


def _run(job: _Job, work: Work) -> None:
    eid = job.export_id
    try:
        with export_job.bind(job.ctx):
            db.mark_export_running(eid)
            work(job.ctx)
        row = db.get_export(eid) or {}
        if row.get("status") not in TERMINAL_STATUSES:
            raise RuntimeError("export worker finished without recording a result")
    except export_job.ExportCancelled:
        log.info("export %s cancelled", eid)
        _finish(eid, lambda: db.mark_export_cancelled(eid))
    except BaseException as e:  # a dead worker must never look alive
        log.exception("export %s failed", eid)
        error = _message(e)
        _finish(eid, lambda: db.mark_export_failed(eid, error))
    finally:
        with _LOCK:
            _ACTIVE.pop(eid, None)
        job.ctx.kill_children()


def _message(e: BaseException) -> str:
    return str(e) or type(e).__name__


def _finish(eid: str, mark: Callable[[], object]) -> None:
    try:
        mark()
    except Exception:
        log.exception("could not record the final state of export %s", eid)


def _shutdown() -> None:
    """Server is exiting: kill every export child so none is orphaned."""
    with _LOCK:
        jobs = list(_ACTIVE.values())
    for job in jobs:
        job.ctx.cancel_event.set()
        job.ctx.kill_children()


atexit.register(_shutdown)
