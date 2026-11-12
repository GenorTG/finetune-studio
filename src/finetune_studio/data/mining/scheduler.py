"""Web side of mining: queue jobs, cancel, reset, and keep a mining worker running while jobs are queued.

Each file is its own job (``job.MiningJob``), a ``mining_jobs`` row. The rows are the queue; the jobs run in the mining
WORKER, a job process of its own (``jobs.kinds.mining`` -> ``worker``), so a web restart (an update, "Apply now",
``fts restart web``) no longer ends them. The worker runs ``MAX_PARALLEL_FILES`` files at once and bounds every helper call
with one semaphore per helper: an API helper serves ``concurrency`` calls at once (Settings -> Helper model, default
``DEFAULT_API_CONCURRENCY``), a local GGUF one (it is one GPU). One worker for all projects keeps that bound global.

The local GGUF is never loaded: submitting jobs needs a helper that is already usable (``resolve_generator``); the job
records which one (``seat.current_target``) and the worker only talks to that. The worker takes no GPU lease: a local
helper is the model already loaded in the inference worker.

After a restart (``recover_after_restart``) jobs whose worker is alive are left alone, jobs whose worker is gone are
``interrupted`` (items and pairs kept; "Fill gaps" continues them) and still-queued jobs get a worker.
"""
from __future__ import annotations

import logging
import threading
import time
from pathlib import Path
from typing import Any

from finetune_studio.data.mining.prompts import PROMPT_VERSION

log = logging.getLogger(__name__)

MAX_PARALLEL_FILES = 16
DEFAULT_API_CONCURRENCY = 8
MAX_API_CONCURRENCY = 32


class MiningError(Exception):
    """A request the scheduler refuses; ``status`` is the HTTP status the route answers with."""

    def __init__(self, message: str, status: int = 409) -> None:
        super().__init__(message)
        self.status = status


def helper_info(job: str = "mining") -> dict[str, Any]:
    """The connection a job runs on, as stored on each mining job: provider, model, kind, reasoning setting, parallel
    calls, prompt version and a tag of the user's extra instructions."""
    from finetune_studio.models.assistant import instructions_tag
    from finetune_studio.models.helper import (
        get_configured_helper_provider,
        get_helper_provider_id,
    )

    row = get_configured_helper_provider(job) or {}
    extra = row.get("extra") or {}
    api = row.get("kind") == "openai_compat"
    concurrency = 1
    if api:
        try:
            concurrency = int(extra.get("concurrency") or DEFAULT_API_CONCURRENCY)
        except (TypeError, ValueError):
            concurrency = DEFAULT_API_CONCURRENCY
        concurrency = max(1, min(MAX_API_CONCURRENCY, concurrency))
    return {
        "provider_id": row.get("id") or get_helper_provider_id(job),
        "label": row.get("label") or row.get("name") or "",
        "kind": row.get("kind") or "",
        "model": row.get("model_id") or "",
        "reasoning_effort": str((extra.get("body") or {}).get("reasoning_effort") or ""),
        "concurrency": concurrency,
        "prompt_version": PROMPT_VERSION,
        "instructions_tag": instructions_tag(job),
    }


KIND = "mining"                       # the worker's job kind (jobs.runner.KINDS)
INTERRUPTED_REASON = "interrupted by a service restart: run Fill gaps to continue"


# ── the worker job ──────────────────────────────────────────────────────────────────────────────────────────────────

def running_mining_job_ids() -> set[str]:
    """Ids of the mining worker jobs that are alive (their files keep running across a web restart)."""
    from finetune_studio.jobs import launch

    return {j["id"] for j in launch.list_jobs(KIND, running_only=True)}


def reconcile(reason: str = INTERRUPTED_REASON) -> int:
    """Running jobs whose worker is gone (crashed, killed, or from before mining ran as a job) become ``interrupted``.

    Reads the rows' workers BEFORE the live ones: a worker that took a job after the first read is alive in the second."""
    from finetune_studio import db

    runners = db.mining.runners_of_running()
    if not runners:
        return 0
    dead = runners - running_mining_job_ids()
    n = sum(db.mining.interrupt_running(reason, runner=r) for r in dead)
    if n:
        log.warning("mining: %d job(s) lost their worker: %s", n, reason)
    return n


def ensure_worker() -> str:
    """A worker that still takes jobs, started when there is none; returns its job id. Call after queueing rows."""
    from finetune_studio.jobs import launch, store

    for j in launch.list_jobs(KIND, running_only=True):
        if (j.get("state") or {}).get("accepting", True):
            return str(j["id"])
    spec = store.JobSpec(id=store.new_id(KIND), kind=KIND, title="mining worker", stop_grace_s=30.0)
    try:
        launch.start(spec)
    except launch.JobRejected as e:
        raise MiningError(f"could not start the mining worker: {e}", e.status if e.status >= 400 else 503) from e
    log.info("mining: started worker %s", spec.id)
    return spec.id


def recover_after_restart() -> dict[str, Any]:
    """Web startup: leave jobs whose worker outlived the restart, interrupt the ones whose worker is gone, and start a
    worker for jobs still queued. Never raises (startup must not fail on it); the reason is logged."""
    from finetune_studio import db

    out: dict[str, Any] = {"interrupted": 0, "queued": 0, "worker": ""}
    out["interrupted"] = reconcile(INTERRUPTED_REASON)
    out["queued"] = sum(1 for j in db.mining.list_jobs(active=True, limit=500) if j["status"] == "queued")
    if out["queued"]:
        try:
            out["worker"] = ensure_worker()
        except MiningError as e:
            log.error("mining: %d queued job(s) have no worker: %s", out["queued"], e)
    return out


def interrupt_after_restart() -> int:
    """Startup (kept for callers of the old name): ``recover_after_restart``; returns how many jobs were interrupted."""
    return int(recover_after_restart()["interrupted"])


def on_finished(job: dict[str, Any]) -> None:
    """Web process (``jobs.watch``): a mining worker ended. Its running files are interrupted with the reason; files
    still queued (submitted while it was exiting or stopping) get a new worker."""
    from finetune_studio import db

    ex = job.get("exit") or {}
    status = str(ex.get("status") or "ended")
    if status not in ("done", "stopped"):
        detail = ex.get("error") or ex.get("detail") or ""
        reconcile(f"the mining worker {status}{': ' + str(detail)[:300] if detail else ''}: run Fill gaps to continue")
    else:
        reconcile(INTERRUPTED_REASON if status == "done" else "the mining worker was stopped: run Fill gaps to continue")
    if any(j["status"] == "queued" for j in db.mining.list_jobs(active=True, limit=500)):
        try:
            ensure_worker()
        except MiningError:
            log.exception("mining: queued jobs have no worker")


class MiningScheduler:
    """The requests the mining routes make (web process); the work runs in the worker job."""

    # ── requests ────────────────────────────────────────────────────────────────────────────────────────────────

    def submit(self, pid: str, source_ids: list[str], kind: str) -> list[dict[str, Any]]:
        """Queue one job per file and make sure a worker runs them. Refuses the whole request (nothing queued) when any
        file cannot run."""
        from finetune_studio import db
        from finetune_studio.data.mining import seat
        from finetune_studio.data.prep.generator import (
            helper_resolution_error,
            resolve_generator,
        )

        if kind not in ("mine", "fill"):
            raise MiningError(f"unknown job kind {kind!r} (mine | fill)", 400)
        ids = list(dict.fromkeys(s for s in source_ids if s))
        if not ids:
            raise MiningError("choose at least one file", 400)
        from finetune_studio.knowledge.sources import ERROR, list_sources

        sources = {s.id: s for s in list_sources(pid)}
        unknown = [s for s in ids if s not in sources]
        if unknown:
            raise MiningError(f"unknown file id(s): {unknown[:10]}", 404)
        reconcile()                                  # a crashed worker's files must not block them as "busy"
        busy = {j["source_id"] for j in db.mining.list_jobs(pid, active=True)}
        has_items = {it["source_id"] for it in db.mining.list_items(pid)}
        problems = []
        for sid in ids:
            name = sources[sid].filename or sid
            if sid in busy:
                problems.append(f"{name}: a job is already working on it")
            elif kind == "mine" and sid in has_items:
                problems.append(f"{name}: already mined (use Fill gaps, or Reset the file to mine it again)")
            elif kind == "fill" and sid not in has_items:
                problems.append(f"{name}: not mined yet (Mine it first)")
            elif sources[sid].status == ERROR:
                problems.append(f"{name}: the parser failed on it ({sources[sid].error or 'no text'})")
        if problems:
            raise MiningError("; ".join(problems))
        if resolve_generator("mining") is None:
            raise MiningError(helper_resolution_error("mining"))
        from finetune_studio.models.assistant import job_instructions

        helper = helper_info("mining")
        helper["target"] = {**seat.current_target(), "instructions": job_instructions("mining")}
        helper["vision"] = {"mode": "off", "local_model": ""}
        if kind == "mine" and self._vision_reader(helper) is not None:     # refused here, before anything is queued
            helper["vision"] = seat.vision_seat()
        jobs = [db.mining.create_job(pid, sid, sources[sid].filename or sid, kind, helper) for sid in ids]
        try:
            ensure_worker()
        except MiningError as e:
            for job in jobs:
                db.mining.update_job(job["id"], status="failed", stage="done", error=str(e), message=f"Failed: {e}")
            raise
        return jobs

    @staticmethod
    def _vision_reader(helper: dict[str, Any]) -> Any:
        """The Vision reader the batch would use, or None when the seat is off. A local reader that would evict a
        different local helper from the one worker in the middle of mining is refused here, before anything is queued.
        (Only a check: the worker builds its own reader from the recorded seat.)"""
        from finetune_studio.models.helper import paths_match
        from finetune_studio.models.vision import Reader, VisionError, resolve_seat

        seat = resolve_seat()
        if seat.mode == "off":
            return None
        if seat.mode == "local" and helper.get("kind") == "local_gguf" and not paths_match(helper.get("model"),
                                                                                          seat.local_model):
            raise MiningError(
                f"the Reading pictures job ({Path(seat.local_model).name}) and the local assistant ({helper.get('model')}) "
                "would share the one inference worker: in Settings -> AI assistant give Reading pictures the same "
                "local model, an API connection, or turn it off")
        try:
            return Reader(seat)
        except VisionError as e:
            raise MiningError(f"Vision reader: {e}") from e

    def cancel(self, pid: str, job_id: str) -> dict[str, Any]:
        """A queued job is cancelled at once; a running one is asked to stop (the worker sees it within a poll)."""
        from finetune_studio import db

        job = db.mining.get_job(job_id)
        if not job or job["project_id"] != pid:
            raise MiningError("unknown job", 404)
        if job["status"] not in db.mining.ACTIVE_STATUSES:
            raise MiningError(f"the job is already {job['status']}")
        if db.mining.cancel_queued(job_id):
            return db.mining.get_job(job_id)  # type: ignore[return-value]
        if not db.mining.request_cancel(job_id):
            job = db.mining.get_job(job_id) or job
            raise MiningError(f"the job is already {job['status']}")
        job = db.mining.get_job(job_id) or job
        if job["status"] == "running" and job.get("runner") not in running_mining_job_ids():
            # Its worker is gone: nothing would ever read the request.
            return db.mining.update_job(job_id, status="cancelled", stage="done", finished_at=time.time(),  # type: ignore[return-value]
                                        message="Cancelled.")
        return job

    def reset(self, pid: str, source_id: str) -> dict[str, int]:
        """Forget a file's inventory, rounds and UNREVIEWED pairs so it can be mined afresh. Reviewed pairs stay."""
        from finetune_studio import db
        from finetune_studio.data.fs import qa as qa_fs

        reconcile()
        if db.mining.list_jobs(pid, source_id=source_id, active=True, limit=1):
            raise MiningError("a job is working on this file: cancel it first")
        removed = 0
        for p in qa_fs.list_qa_pairs(pid, source_id=source_id, status="pending"):
            removed += int(qa_fs.delete_qa_pair(pid, str(p["id"])))
        items = len(db.mining.list_items(pid, source_id))
        db.mining.delete_file_records(pid, source_id)
        return {"pending_pairs_removed": removed, "items_removed": items}


_scheduler: MiningScheduler | None = None
_scheduler_lock = threading.Lock()


def get_scheduler() -> MiningScheduler:
    global _scheduler
    with _scheduler_lock:
        if _scheduler is None:
            _scheduler = MiningScheduler()
        return _scheduler
