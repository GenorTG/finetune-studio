"""Process-wide queue of mining jobs: files in parallel, helper calls bounded globally.

Each file is its own job (``job.MiningJob``). Jobs run on a thread pool of ``MAX_PARALLEL_FILES``; every helper call of every
job goes through one semaphore per helper, sized by the helper: an API helper serves ``concurrency`` calls at once
(Settings -> Helper model, default ``DEFAULT_API_CONCURRENCY``), a local GGUF one (it is one GPU). So a 19-file corpus on an API
helper runs as one wide stream of calls, and on the local GGUF as one call at a time.

The local GGUF is never loaded here: submitting jobs needs a helper that is already usable (``resolve_generator``); an API
helper is seated on demand by the generator. A restart cannot resume a thread, so jobs that were running are marked
``interrupted`` at startup (``interrupt_after_restart``); their items and pairs are kept, and "Fill gaps" continues them.
"""
from __future__ import annotations

import logging
import threading
from concurrent.futures import ThreadPoolExecutor
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


def helper_info() -> dict[str, Any]:
    """The seated helper as stored on each job: provider, model, kind, reasoning setting, parallel calls, prompt version."""
    from finetune_studio.models.helper import (
        get_configured_helper_provider,
        get_helper_provider_id,
    )

    row = get_configured_helper_provider() or {}
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
        "provider_id": row.get("id") or get_helper_provider_id(),
        "label": row.get("label") or row.get("name") or "",
        "kind": row.get("kind") or "",
        "model": row.get("model_id") or "",
        "reasoning_effort": str((extra.get("body") or {}).get("reasoning_effort") or ""),
        "concurrency": concurrency,
        "prompt_version": PROMPT_VERSION,
    }


class _Batch:
    """The files of one submit: the last one to finish closes the shared Vision reader (unloads a local model it loaded)."""

    def __init__(self, n: int, vision: Any) -> None:
        self.vision = vision
        self._left = n
        self._lock = threading.Lock()

    def done(self) -> None:
        with self._lock:
            self._left -= 1
            last = self._left == 0
        if last and self.vision is not None:
            self.vision.close()


class MiningScheduler:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._pool = ThreadPoolExecutor(max_workers=MAX_PARALLEL_FILES, thread_name_prefix="fts-mining")
        self._limiters: dict[tuple[str, int], threading.BoundedSemaphore] = {}
        self._cancel: dict[str, threading.Event] = {}

    # ── requests ────────────────────────────────────────────────────────────────────────────────────────────────

    def submit(self, pid: str, source_ids: list[str], kind: str) -> list[dict[str, Any]]:
        """Queue one job per file. Refuses the whole request (nothing queued) when any file cannot run."""
        from finetune_studio import db
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
        chat = resolve_generator()
        if chat is None:
            raise MiningError(helper_resolution_error())
        helper = helper_info()
        limiter = self._limiter(helper["provider_id"], helper["concurrency"])
        vision = self._vision_reader(helper) if kind == "mine" else None
        batch = _Batch(len(ids), vision)
        jobs = []
        for sid in ids:
            job = db.mining.create_job(pid, sid, sources[sid].filename or sid, kind, helper)
            event = threading.Event()
            with self._lock:
                self._cancel[job["id"]] = event
            self._pool.submit(self._run, pid, job, chat, limiter, helper, event, batch)
            jobs.append(job)
        return jobs

    @staticmethod
    def _vision_reader(helper: dict[str, Any]) -> Any:
        """The batch's Vision reader, or None when the seat is off. A local reader that would evict a different local
        helper from the one worker in the middle of mining is refused here, before anything is queued."""
        from finetune_studio.models.helper import paths_match
        from finetune_studio.models.vision import Reader, VisionError, get_seat

        seat = get_seat()
        if seat.mode == "off":
            return None
        if seat.mode == "local" and helper.get("kind") == "local_gguf" and not paths_match(helper.get("model"),
                                                                                          seat.local_model):
            raise MiningError(
                f"the Vision reader ({Path(seat.local_model).name}) and the local helper ({helper.get('model')}) would "
                "share the one inference worker: pick the helper's own GGUF as the Vision reader, use an API vision "
                "model, or turn the Vision reader off")
        try:
            return Reader(seat)
        except VisionError as e:
            raise MiningError(f"Vision reader: {e}") from e

    def cancel(self, pid: str, job_id: str) -> dict[str, Any]:
        from finetune_studio import db

        job = db.mining.get_job(job_id)
        if not job or job["project_id"] != pid:
            raise MiningError("unknown job", 404)
        if job["status"] not in db.mining.ACTIVE_STATUSES:
            raise MiningError(f"the job is already {job['status']}")
        with self._lock:
            event = self._cancel.get(job_id)
        if event is None:                       # queued before a restart and never picked up: nothing runs it
            return db.mining.update_job(job_id, status="cancelled", stage="done", message="Cancelled.")  # type: ignore[return-value]
        event.set()
        return db.mining.update_job(job_id, message="Cancelling…")  # type: ignore[return-value]

    def reset(self, pid: str, source_id: str) -> dict[str, int]:
        """Forget a file's inventory, rounds and UNREVIEWED pairs so it can be mined afresh. Reviewed pairs stay."""
        from finetune_studio import db
        from finetune_studio.data.fs import qa as qa_fs

        if db.mining.list_jobs(pid, source_id=source_id, active=True, limit=1):
            raise MiningError("a job is working on this file: cancel it first")
        removed = 0
        for p in qa_fs.list_qa_pairs(pid, source_id=source_id, status="pending"):
            removed += int(qa_fs.delete_qa_pair(pid, str(p["id"])))
        items = len(db.mining.list_items(pid, source_id))
        db.mining.delete_file_records(pid, source_id)
        return {"pending_pairs_removed": removed, "items_removed": items}

    # ── execution ───────────────────────────────────────────────────────────────────────────────────────────────

    def _limiter(self, provider_id: str, concurrency: int) -> threading.BoundedSemaphore:
        with self._lock:
            key = (provider_id, concurrency)
            if key not in self._limiters:
                self._limiters[key] = threading.BoundedSemaphore(concurrency)
            return self._limiters[key]

    def _run(self, pid: str, job: dict[str, Any], chat: Any, limiter: threading.BoundedSemaphore,
             helper: dict[str, Any], cancel: threading.Event, batch: _Batch | None = None) -> None:
        from finetune_studio.data.mining.job import JobContext, MiningJob

        try:
            if cancel.is_set():
                from finetune_studio import db
                db.mining.update_job(job["id"], status="cancelled", stage="done", message="Cancelled.")
                return
            MiningJob(JobContext(pid=pid, job=job, chat=chat, limiter=limiter, model=helper["model"],
                                 workers=helper["concurrency"], cancel=cancel,
                                 vision=batch.vision if batch else None)).run()
        finally:
            with self._lock:
                self._cancel.pop(job["id"], None)
            if batch is not None:
                batch.done()


_scheduler: MiningScheduler | None = None
_scheduler_lock = threading.Lock()


def get_scheduler() -> MiningScheduler:
    global _scheduler
    with _scheduler_lock:
        if _scheduler is None:
            _scheduler = MiningScheduler()
        return _scheduler


def interrupt_after_restart() -> int:
    """Startup: jobs that claimed to run belonged to the previous process. Their items and pairs stay."""
    from finetune_studio import db

    n = db.mining.interrupt_active_jobs("interrupted by a service restart: run Fill gaps to continue")
    if n:
        log.warning("mining: %d job(s) were interrupted by the restart", n)
    return n
