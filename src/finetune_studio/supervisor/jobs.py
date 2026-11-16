"""One-shot job processes owned by the supervisor (unlike components they are never restarted).

A job outlives web restarts and crashes because the supervisor, not the web app, is its parent. GPU lease: at most
one ``gpu`` job runs at a time, and the inference worker's model is unloaded before one starts.

A ``gpu`` job whose spec sets ``queue_gpu`` (the pictures job an upload starts) is not refused while the lease is
held: it waits in this table's queue (``queued.json`` says behind what, for the pages) and starts, in order, once
the holder ends or releases the lease. Every other GPU job is still refused with ``JobBusy`` (409).
"""
from __future__ import annotations

import asyncio
import contextlib
import os
import signal
import sys
import time
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from finetune_studio.jobs import store
from finetune_studio.supervisor.env import child_env
from finetune_studio.supervisor.events import EventLog

LOG_RING = 300
TAIL_LINES = 25
# How long a GPU job's start waits for the inference worker to unload its model (it queues behind a load).
UNLOAD_TIMEOUT_S = 60.0
# How often a waiting GPU job checks whether the lease is free (the holder's exit or its ``gpu_released`` mark).
QUEUE_POLL_S = 1.0


class JobBusy(RuntimeError):
    """Another GPU job holds the lease."""


class UnknownJob(KeyError):
    pass


@dataclass
class _Job:
    spec: store.JobSpec
    proc: asyncio.subprocess.Process
    started_wall: float
    logs: deque[str] = field(default_factory=lambda: deque(maxlen=LOG_RING))
    stop_requested: bool = False
    waiter: asyncio.Task[None] | None = None


def _describe(returncode: int) -> dict[str, Any]:
    if returncode < 0:
        try:
            return {"code": None, "signal": signal.Signals(-returncode).name}
        except ValueError:
            return {"code": None, "signal": f"SIG{-returncode}"}
    return {"code": returncode, "signal": None}


class JobTable:
    def __init__(self, events: EventLog, *, cwd: str | None = None, inference_socket: Path | None = None) -> None:
        self.events = events
        self.cwd = cwd
        self.inference_socket = inference_socket
        self._jobs: dict[str, _Job] = {}
        # Jobs between "start asked" and "process spawned" (a GPU job waits there for the inference unload, which
        # can sit behind a model load for tens of seconds): they hold the lease too, so a second start in that
        # window is refused instead of running two GPU jobs at once (two builds of one project, 2026-11-13).
        self._starting: dict[str, store.JobSpec] = {}
        # GPU jobs waiting for the lease (``JobSpec.queue_gpu``), oldest first.
        self._queued: dict[str, store.JobSpec] = {}
        self._drain_task: asyncio.Task[None] | None = None

    def running(self) -> list[_Job]:
        return [j for j in self._jobs.values() if j.proc.returncode is None]

    def gpu_holder(self) -> str:
        """Who holds the GPU lease, in words ("" = free): a running GPU job, or one between start and spawn."""
        holder = next((j for j in self.running() if j.spec.gpu and not store.gpu_released(j.spec.id)), None)
        if holder:
            return f"{holder.spec.kind} job {holder.spec.id} ({holder.spec.title or 'untitled'})"
        pending = next((s for s in self._starting.values() if s.gpu), None)
        if pending is not None:
            return f"{pending.kind} job {pending.id} ({pending.title or 'untitled'}), which is starting"
        return ""

    async def start(self, job_id: str) -> dict[str, Any]:
        spec = store.read_spec(job_id)            # FileNotFoundError / ValueError -> caller answers 404/400
        if job_id in self._jobs and self._jobs[job_id].proc.returncode is None:
            return self.describe(job_id)
        if job_id in self._starting:
            raise JobBusy(f"job {job_id} is already starting")
        if job_id in self._queued:
            return self.describe(job_id)
        if spec.gpu:
            holder = self.gpu_holder()
            if holder and spec.queue_gpu:
                return self._enqueue(spec, holder)
            if holder:
                raise JobBusy(f"the GPU is busy with {holder}; wait for it or stop it")
        self._starting[job_id] = spec
        try:
            if spec.gpu:
                await self._unload_inference_model()
            proc = await asyncio.create_subprocess_exec(
                sys.executable, "-m", "finetune_studio.jobs.runner", "--id", job_id,
                cwd=self.cwd, env=child_env(), stdin=asyncio.subprocess.DEVNULL,
                stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.STDOUT, start_new_session=True,
            )
        finally:
            self._starting.pop(job_id, None)
        job = _Job(spec=spec, proc=proc, started_wall=time.time())
        self._jobs[job_id] = job
        store.write_pid(job_id, proc.pid)
        store.clear_queued(job_id)
        job.waiter = asyncio.create_task(self._watch(job), name=f"job-{job_id}")
        self.events.emit("jobs", "job_started", job=job_id, job_kind=spec.kind, pid=proc.pid, gpu=spec.gpu)
        return self.describe(job_id)

    def _enqueue(self, spec: store.JobSpec, holder: str) -> dict[str, Any]:
        self._queued[spec.id] = spec
        store.write_queued(spec.id, {"waiting_for": holder, "owner_pid": os.getpid()})
        self.events.emit("jobs", "job_queued_for_gpu", job=spec.id, job_kind=spec.kind, waiting_for=holder)
        if self._drain_task is None or self._drain_task.done():
            self._drain_task = asyncio.create_task(self._drain(), name="jobs-gpu-queue")
        return self.describe(spec.id)

    async def _drain(self) -> None:
        """Start the waiting GPU jobs one by one, each once the lease is free."""
        while self._queued:
            await asyncio.sleep(QUEUE_POLL_S)
            holder = self.gpu_holder()
            if holder:
                for jid in self._queued:              # the pages say behind what it waits (that can change)
                    rec = store.read_queued(jid) or {}
                    if rec.get("waiting_for") != holder:
                        store.write_queued(jid, {**rec, "waiting_for": holder, "owner_pid": os.getpid()})
                continue
            job_id = next(iter(self._queued))
            self._queued.pop(job_id)
            # The record stays until the process is spawned (``start`` clears it): the GPU job's start first
            # waits for the inference model to unload, and the pages must not read that minute as "not running".
            rec = store.read_queued(job_id) or {}
            store.write_queued(job_id, {**rec, "waiting_for": "the GPU to be freed (unloading the inference model)",
                                        "owner_pid": os.getpid()})
            self.events.emit("jobs", "job_dequeued_for_gpu", job=job_id)
            try:
                await self.start(job_id)
            except Exception as e:  # noqa: BLE001 - the job failed to start: record it, keep draining the rest
                store.clear_queued(job_id)
                store.write_exit(job_id, {"status": "failed", "error": f"not started after waiting for the GPU: {e}",
                                          "code": None})
                self.events.emit("jobs", "job_start_failed", job=job_id, error=str(e)[:200])

    async def _unload_inference_model(self) -> None:
        """GPU lease: the inference worker's model goes before a GPU job loads its own."""
        sock = self.inference_socket
        if sock is None or not sock.exists():
            return
        from finetune_studio.supervisor.client import UnixHTTPConnection

        def _post() -> int:
            conn = UnixHTTPConnection(str(sock), UNLOAD_TIMEOUT_S)
            try:
                conn.request("POST", "/v1/unload")
                return conn.getresponse().status
            finally:
                conn.close()
        try:
            status = await asyncio.to_thread(_post)
            self.events.emit("jobs", "inference_unloaded_for_gpu_job", status=status)
        except OSError as e:
            self.events.emit("jobs", "inference_unload_failed", error=str(e)[:200])

    async def _watch(self, job: _Job) -> None:
        assert job.proc.stdout is not None
        async for raw in job.proc.stdout:
            line = raw.decode("utf-8", "replace").rstrip("\n")
            job.logs.append(line)
            sys.stdout.write(f"[job {job.spec.id}] {line}\n")
            sys.stdout.flush()
        rc = await job.proc.wait()
        info = _describe(rc)
        tail = list(job.logs)[-TAIL_LINES:]
        # The runner writes its own record on any normal end; this one only lands after a hard death.
        store.write_exit(job.spec.id, {"status": "stopped" if job.stop_requested else "crashed", **info,
                                       "tail": tail, "ran_s": round(time.time() - job.started_wall, 1)})
        self.events.emit("jobs", "job_exited", job=job.spec.id, job_kind=job.spec.kind, **info,
                         stop_requested=job.stop_requested, tail=[] if rc == 0 else tail)

    async def stop(self, job_id: str) -> dict[str, Any]:
        if job_id in self._queued:
            self._queued.pop(job_id)
            store.clear_queued(job_id)
            store.write_exit(job_id, {"status": "stopped", "detail": "stopped while waiting for the GPU",
                                      "code": None})
            self.events.emit("jobs", "job_stop_requested", job=job_id, queued=True)
            return self.describe(job_id)
        job = self._get(job_id)
        if job.proc.returncode is not None:
            return self.describe(job_id)
        job.stop_requested = True
        self.events.emit("jobs", "job_stop_requested", job=job_id)
        with contextlib.suppress(ProcessLookupError):
            os.killpg(job.proc.pid, signal.SIGTERM)
        asyncio.create_task(self._escalate(job), name=f"job-stop-{job_id}")
        return self.describe(job_id)

    async def _escalate(self, job: _Job) -> None:
        try:
            await asyncio.wait_for(job.proc.wait(), job.spec.stop_grace_s)
        except TimeoutError:
            self.events.emit("jobs", "job_kill_escalation", job=job.spec.id, after_s=job.spec.stop_grace_s)
            with contextlib.suppress(ProcessLookupError):
                os.killpg(job.proc.pid, signal.SIGKILL)

    async def shutdown(self) -> None:
        """Supervisor going down: give every running job its grace, then kill."""
        for job_id in list(self._queued):
            await self.stop(job_id)
        if self._drain_task is not None:
            self._drain_task.cancel()
        for job in self.running():
            await self.stop(job.spec.id)
        for job in list(self._jobs.values()):
            if job.waiter:
                with contextlib.suppress(Exception):
                    await asyncio.wait_for(job.waiter, job.spec.stop_grace_s + 10)

    def _get(self, job_id: str) -> _Job:
        try:
            return self._jobs[job_id]
        except KeyError:
            raise UnknownJob(job_id) from None

    def describe(self, job_id: str) -> dict[str, Any]:
        if job_id in self._queued:
            spec = self._queued[job_id]
            rec = store.read_queued(job_id) or {}
            return {"id": job_id, "kind": spec.kind, "title": spec.title, "project_id": spec.project_id,
                    "gpu": spec.gpu, "pid": None, "running": True, "queued": True,
                    "waiting_for": rec.get("waiting_for") or "", "uptime_s": None, "stop_requested": False,
                    "exit": None}
        if job_id not in self._jobs:
            try:
                spec = store.read_spec(job_id)
            except (FileNotFoundError, ValueError):
                spec = None
            ex = store.read_exit(job_id) if spec is not None else None
            if spec is not None and ex is not None and spec.queue_gpu:
                # Stopped (or failed to start) while it waited: it never had a process.
                return {"id": job_id, "kind": spec.kind, "title": spec.title, "project_id": spec.project_id,
                        "gpu": spec.gpu, "pid": None, "running": False, "queued": False, "uptime_s": None,
                        "stop_requested": ex.get("status") == "stopped", "exit": ex}
        job = self._get(job_id)
        running = job.proc.returncode is None
        return {"id": job_id, "kind": job.spec.kind, "title": job.spec.title, "project_id": job.spec.project_id,
                "gpu": job.spec.gpu, "pid": job.proc.pid, "running": running, "queued": False,
                "uptime_s": round(time.time() - job.started_wall, 1) if running else None,
                "stop_requested": job.stop_requested,
                "exit": None if running else store.read_exit(job_id)}

    def logs(self, job_id: str, lines: int = 100) -> list[str]:
        if job_id in self._queued:
            return []
        return list(self._get(job_id).logs)[-lines:]

    def snapshot(self) -> list[dict[str, Any]]:
        return [self.describe(j) for j in [*self._jobs, *self._queued]]
