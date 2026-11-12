"""The mining worker: the job process (``jobs.kinds.mining``) that runs queued mining jobs, so a web restart no longer ends them.

The ``mining_jobs`` rows are the queue: the web app inserts them ``queued`` and makes sure a worker runs
(``scheduler.ensure_worker``); the worker takes them (``db.mining.claim_queued``, recording its job id as ``runner``),
runs up to ``MAX_PARALLEL_FILES`` files at once and bounds every helper call with one semaphore per helper (an API
helper: its ``concurrency``; a local GGUF: 1). One worker for every project keeps those bounds global.

Each job carries the helper the web app resolved at submit (``helper["target"]``, ``seat``) and the Vision reader seat
(``helper["vision"]``). A local helper is only used while it is the model loaded in the inference worker; this process
never loads one. A local Vision reader is loaded in the inference worker on first use and unloaded once the queue is
empty (as a batch did before).

Cancel: the web app writes ``cancel_requested_at`` on the row; the worker polls and sets that job's cancel flag.
Stopping the worker (SIGTERM: Jobs page, ``fts down``) stops its running files at their next helper call and records
them ``interrupted`` (items and pairs kept; Fill gaps continues them); queued files stay queued for the next worker.
With nothing left to do the worker publishes ``accepting: false``, takes one last look at the queue and exits; a submit
that saw ``accepting: false`` starts a new worker, so a file queued at that moment is never left without one.
"""
from __future__ import annotations

import logging
import threading
import time
from collections.abc import Callable
from concurrent.futures import Future, ThreadPoolExecutor
from typing import Any

from finetune_studio.data.mining import seat
from finetune_studio.data.mining.scheduler import (
    DEFAULT_API_CONCURRENCY,
    MAX_API_CONCURRENCY,
    MAX_PARALLEL_FILES,
)

log = logging.getLogger(__name__)

POLL_S = 0.25
IDLE_EXIT_S = 2.0
STOPPED_REASON = "the mining worker was stopped: run Fill gaps to continue"

Publish = Callable[[dict[str, Any], bool], None]


def _concurrency(helper: dict[str, Any]) -> int:
    try:
        n = int(helper.get("concurrency") or (DEFAULT_API_CONCURRENCY if helper.get("kind") == "openai_compat" else 1))
    except (TypeError, ValueError):
        n = 1
    return max(1, min(MAX_API_CONCURRENCY, n))


def target_of(helper: dict[str, Any]) -> dict[str, Any]:
    """The recorded target; jobs queued before targets were recorded are derived from their helper fields."""
    target = helper.get("target")
    if isinstance(target, dict) and target.get("kind"):
        return target
    if helper.get("kind") == "openai_compat":
        return {"kind": "api", "provider_id": str(helper.get("provider_id") or "")}
    return {"kind": "local", "model_path": str(helper.get("model") or "")}


class Worker:
    def __init__(self, runner: str, stop: threading.Event, publish: Publish) -> None:
        from finetune_studio.db import connection

        self.runner = runner
        self.stop = stop
        self._publish = publish
        self._db_path = connection.settings.db_path
        self._pool = ThreadPoolExecutor(max_workers=MAX_PARALLEL_FILES, thread_name_prefix="fts-mining")
        self._lock = threading.Lock()
        self._limiters: dict[tuple[str, int], threading.BoundedSemaphore] = {}
        self._readers: dict[str, Any] = {}
        self._running: dict[str, tuple[Future[None], threading.Event]] = {}
        self._user_cancelled: set[str] = set()
        self._stopping = False
        self.finished = 0
        self._last_state: dict[str, Any] | None = None

    # ── loop ────────────────────────────────────────────────────────────────────────────────────────────────────

    def run(self) -> dict[str, Any]:
        self._state(accepting=True)
        idle_since: float | None = None
        try:
            while not self.stop.is_set() and self._same_db():
                self._reap()
                self._poll_cancels()
                claimed = self._claim()
                if self._running or claimed:
                    idle_since = None
                    self._state(accepting=True)
                    self.stop.wait(POLL_S)
                    continue
                self._close_readers()
                now = time.monotonic()
                idle_since = idle_since or now
                if now - idle_since >= IDLE_EXIT_S:
                    self._state(accepting=False)        # a submit from here on starts a new worker...
                    if not self._claim():               # ...and one queued just before is still seen here
                        break
                    idle_since = None
                    self._state(accepting=True)
                self.stop.wait(POLL_S)
        finally:
            self._finish()
        return {"status": "stopped" if self._stopping else "done", "files": self.finished}

    def _same_db(self) -> bool:
        """The worker serves the DB it started on (its process never changes it; an in-process test run does)."""
        from finetune_studio.db import connection
        return connection.settings.db_path == self._db_path

    def _state(self, *, accepting: bool) -> None:
        with self._lock:
            state = {"accepting": accepting, "running": sorted(self._running), "finished": self.finished}
        if state != self._last_state:                       # the pages and ensure_worker read it; write on change only
            self._last_state = state
            self._publish(state, True)

    def _claim(self) -> list[dict[str, Any]]:
        from finetune_studio import db

        with self._lock:
            free = MAX_PARALLEL_FILES - len(self._running)
        jobs = db.mining.claim_queued(self.runner, free)
        for job in jobs:
            event = threading.Event()
            with self._lock:
                self._running[job["id"]] = (self._pool.submit(self._run_one, job, event), event)
        return jobs

    def _reap(self) -> None:
        with self._lock:
            done = [jid for jid, (fut, _) in self._running.items() if fut.done()]
            for jid in done:
                self._running.pop(jid)
                self._user_cancelled.discard(jid)
            self.finished += len(done)

    def _poll_cancels(self) -> None:
        from finetune_studio import db

        with self._lock:
            ids = list(self._running)
        for jid in db.mining.cancel_requested(ids):
            with self._lock:
                entry = self._running.get(jid)
                if entry is not None and not entry[1].is_set():
                    self._user_cancelled.add(jid)
                    entry[1].set()

    def _finish(self) -> None:
        from finetune_studio import db

        if self.stop.is_set():
            self._stopping = True
            with self._lock:
                for _, event in self._running.values():
                    event.set()
        self._pool.shutdown(wait=True, cancel_futures=self._stopping)   # files not started yet stay 'running' ...
        self._reap()
        if self._stopping:
            n = db.mining.interrupt_running(STOPPED_REASON, runner=self.runner)     # ... and are interrupted here
            if n:
                log.warning("mining worker %s stopped: %d file(s) interrupted", self.runner, n)
        self._close_readers()
        seat.forget_providers()

    # ── one file ────────────────────────────────────────────────────────────────────────────────────────────────

    def _run_one(self, job: dict[str, Any], cancel: threading.Event) -> None:
        from finetune_studio import db
        from finetune_studio.data.mining.job import JobContext, MiningJob

        helper = job.get("helper") or {}
        try:
            chat = seat.chat_for(target_of(helper))
            concurrency = _concurrency(helper)
            limiter = self._limiter(str(helper.get("provider_id") or ""), concurrency)
            vision = self._reader(helper.get("vision")) if job["kind"] == "mine" else None
        except Exception as exc:  # noqa: BLE001 - the reason is the job's error; the worker goes on with the others
            log.warning("mining job %s cannot start: %s", job["id"], exc)
            db.mining.update_job(job["id"], status="failed", stage="done", finished_at=time.time(),
                                 error=str(exc)[:2000], message=f"Failed: {exc}")
            return
        MiningJob(JobContext(pid=job["project_id"], job=job, chat=chat, limiter=limiter, model=str(helper.get("model") or ""),
                             workers=concurrency, cancel=cancel, vision=vision)).run()
        with self._lock:
            by_user = job["id"] in self._user_cancelled
        if cancel.is_set() and not by_user and self.stop.is_set():
            # Stopped with the worker, not by the reviewer: say so (Fill gaps continues it), never "cancelled".
            db.mining.update_job(job["id"], status="interrupted", error=STOPPED_REASON, finished_at=time.time(),
                                 message="Interrupted: the mining worker was stopped.")

    def _limiter(self, provider_id: str, concurrency: int) -> threading.BoundedSemaphore:
        with self._lock:
            key = (provider_id, concurrency)
            if key not in self._limiters:
                self._limiters[key] = threading.BoundedSemaphore(concurrency)
            return self._limiters[key]

    def _reader(self, recorded: Any) -> Any:
        """The Vision reader for the seat recorded on the job (None when off), shared by every file that names it."""
        from finetune_studio.models.vision import Reader, Seat, VisionError

        if not isinstance(recorded, dict):
            recorded = seat.vision_seat()                  # queued before the seat was recorded on jobs
        s = Seat.from_dict(recorded)
        if s.mode == "off":
            return None
        with self._lock:
            if s.as_json() not in self._readers:
                try:
                    self._readers[s.as_json()] = Reader(s)
                except VisionError as e:
                    raise RuntimeError(f"Vision reader: {e}") from e
            return self._readers[s.as_json()]

    def _close_readers(self) -> None:
        with self._lock:
            readers, self._readers = list(self._readers.values()), {}
        for reader in readers:
            reader.close()                                  # unloads a local model it loaded


def run(runner: str, stop: threading.Event, publish: Publish) -> dict[str, Any]:
    return Worker(runner, stop, publish).run()
