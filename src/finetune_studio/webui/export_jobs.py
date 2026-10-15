"""Lifecycle of WebUI export jobs: one job process, one row, one at a time.

An export (merge, GGUF convert, quantize, abliterate) takes minutes and saturates the GPU or CPU, so it runs as a
supervised job process (``jobs.kinds.export``): a web restart (an update, "Apply now" in Settings) no longer ends
it. ``start_job`` creates the ``model_exports`` row (status ``queued``) and launches job ``export-<export id>``; the
row then carries the real state — ``status``, ``phase`` + ``phase_detail`` (reported by the pipeline through
:mod:`finetune_studio.training.export_job`), a ``heartbeat_at`` timestamp and the error text — so a page reload,
a second tab, the SSE stream and a restarted web app all read the same truth.

The work crosses the process boundary as a :data:`WorkSpec`: the name of an ``export_work`` factory and its
JSON-safe keyword arguments; the job process builds the callable.

Concurrency policy: **single flight, 409 on conflict.** Every export shares the one GPU and one llama.cpp
toolchain; a second export while one is queued or running is refused with :class:`ExportBusy` (HTTP 409 naming the
active job), and so is an export while another GPU job (training) holds the card.

Cancel stops a job at its next checkpoint: immediately while a llama.cpp child runs (its whole process group is
killed), but only between phases while the in-process adapter merge is running (a torch merge cannot be
interrupted).
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from typing import Any

from finetune_studio import db
from finetune_studio.jobs import launch, store
from finetune_studio.training import export_job

log = logging.getLogger(__name__)

TERMINAL_STATUSES = frozenset({"done", "failed", "error", "cancelled"})
KIND = "export"
STOP_GRACE_S = 120.0                 # a torch merge only stops between phases

# ``work`` runs inside the bound ExportContext in the job process. It does the export, finishes every row it owns
# with ``db.mark_export_done`` and raises on failure (the message becomes the row's ``error``).
Work = Callable[[export_job.ExportContext], None]
# (``export_work`` factory name, its keyword arguments): what start_job sends to the job process.
WorkSpec = tuple[str, dict[str, Any]]
FACTORIES = frozenset({"multi_export_work", "single_quant_work", "imatrix_work"})


class ExportBusy(RuntimeError):
    """Another export is queued or running, or another GPU job holds the card (single-flight policy)."""

    def __init__(self, active: dict, message: str | None = None) -> None:
        super().__init__(message or (
            f"another export is already {active.get('status') or 'running'} (id {active.get('id')})"))
        self.active = active


def job_id(export_id: str) -> str:
    return f"{KIND}-{export_id}"


def _running() -> list[str]:
    """Export ids whose job process is alive."""
    out = []
    for j in launch.list_jobs(KIND, running_only=True):
        try:
            out.append(store.read_spec(j["id"]).params["export_id"])
        except (FileNotFoundError, ValueError, KeyError):
            continue
    return out


def active_ids() -> list[str]:
    return _running()


def is_active(export_id: str) -> bool:
    try:
        return launch.describe(job_id(export_id))["running"]
    except (FileNotFoundError, ValueError):
        return False


def start_job(*, project_id: str, run_id: str, fmt: str, quant: str, quants: list[str] | None,
              work: WorkSpec) -> dict:
    """Create the export row and launch its job; return the queued row.

    Raises :class:`ExportBusy` when an export is in flight or the GPU is held by another job.
    """
    factory, kwargs = work
    if factory not in FACTORIES:
        raise ValueError(f"unknown export work {factory!r}")
    for active_id in _running():
        active = db.get_export(active_id) or {"id": active_id, "status": "running"}
        # A job whose row is already terminal only has its process epilogue left: the UI sees "done" and
        # immediately posts the next selected format, which must not bounce off a 409.
        if active.get("status") not in TERMINAL_STATUSES:
            raise ExportBusy(active)
    row = db.create_export(project_id=project_id, run_id=run_id, format=fmt, quant=quant, quants=quants)
    eid = row["id"]
    spec = store.JobSpec(
        id=job_id(eid), kind=KIND, gpu=True, stop_grace_s=STOP_GRACE_S, project_id=project_id or None,
        title=f"export {fmt} {quant}".strip(),
        params={"export_id": eid, "factory": factory, "kwargs": kwargs},
    )
    try:
        launch.start(spec)
    except launch.JobRejected as e:
        db.mark_export_failed(eid, f"not started: {e}")
        raise ExportBusy({"id": None, "status": "running", "kind": "gpu"}, str(e)) from e
    return row


def cancel_job(export_id: str) -> bool:
    """Ask a running export to stop; False when it is not running or already finished.

    The job is another process: "cancelling" is written only while the row is still active (one SQL statement),
    so a cancel can never report success for, or write over, a finished export."""
    if not is_active(export_id):
        return False
    if not db.set_export_phase_if_active(export_id, "cancelling", "stopping…"):
        return False
    launch.stop(job_id(export_id))     # SIGTERM -> the job sets its cancel flag; children are killed there
    return True


def run_in_job(export_id: str, work: Work, stop: Any, release_gpu: Callable[[], None] = lambda: None) -> str:
    """The job process side: run ``work`` with its ExportContext bound; returns done | cancelled | failed.

    ``stop`` (the job's stop event) is the export's cancel flag. The final state is written only after the
    export's children are dead; then ``release_gpu`` lets the next selected format start at once."""
    ctx = export_job.ExportContext(
        export_id,
        on_phase=lambda phase, detail: db.set_export_phase(export_id, phase, detail),
        on_heartbeat=lambda: db.heartbeat_export(export_id),
    )
    ctx.cancel_event = stop
    final: Callable[[], object] | None = None
    outcome = "done"
    try:
        with export_job.bind(ctx):
            db.mark_export_running(export_id)
            work(ctx)
        row = db.get_export(export_id) or {}
        if row.get("status") not in TERMINAL_STATUSES:
            raise RuntimeError("export worker finished without recording a result")
    except export_job.ExportCancelled:
        log.info("export %s cancelled", export_id)
        outcome, final = "cancelled", lambda: db.mark_export_cancelled(export_id)
    except BaseException as e:  # a dead worker must never look alive
        log.exception("export %s failed", export_id)
        error = str(e) or type(e).__name__
        outcome, final = "failed", lambda: db.mark_export_failed(export_id, error)
    finally:
        ctx.kill_children()             # "cancelled"/"failed" is only written once its processes are gone
        if final is not None:
            try:
                final()
            except Exception:
                log.exception("could not record the final state of export %s", export_id)
        release_gpu()
    if outcome == "done":
        from finetune_studio.webui.export_work import refresh_registry_quietly
        refresh_registry_quietly()      # web process only (tests); a job process leaves it to on_finished
    return outcome


def running_export_ids() -> set[str]:
    """For the startup reconcile: rows whose job outlived the web restart are left alone."""
    return set(_running())


def on_finished(job: dict[str, Any]) -> None:
    """Web process (``jobs.watch``): an export job ended; make its files visible to Chat/Inference."""
    from finetune_studio.webui.export_work import refresh_registry_quietly
    refresh_registry_quietly()
