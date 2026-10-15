"""One-shot job processes owned by the supervisor (unlike components they are never restarted).

A job outlives web restarts and crashes because the supervisor, not the web app, is its parent. GPU lease: at most
one ``gpu`` job runs at a time, and the inference worker's model is unloaded before one starts.
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

    def running(self) -> list[_Job]:
        return [j for j in self._jobs.values() if j.proc.returncode is None]

    async def start(self, job_id: str) -> dict[str, Any]:
        spec = store.read_spec(job_id)            # FileNotFoundError / ValueError -> caller answers 404/400
        if job_id in self._jobs and self._jobs[job_id].proc.returncode is None:
            return self.describe(job_id)
        if spec.gpu:
            holder = next((j for j in self.running() if j.spec.gpu), None)
            if holder:
                raise JobBusy(f"the GPU is busy with {holder.spec.kind} job {holder.spec.id} "
                              f"({holder.spec.title or 'untitled'}); wait for it or stop it")
            await self._unload_inference_model()
        proc = await asyncio.create_subprocess_exec(
            sys.executable, "-m", "finetune_studio.jobs.runner", "--id", job_id,
            cwd=self.cwd, env=child_env(), stdin=asyncio.subprocess.DEVNULL,
            stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.STDOUT, start_new_session=True,
        )
        job = _Job(spec=spec, proc=proc, started_wall=time.time())
        self._jobs[job_id] = job
        store.write_pid(job_id, proc.pid)
        job.waiter = asyncio.create_task(self._watch(job), name=f"job-{job_id}")
        self.events.emit("jobs", "job_started", job=job_id, job_kind=spec.kind, pid=proc.pid, gpu=spec.gpu)
        return self.describe(job_id)

    async def _unload_inference_model(self) -> None:
        """GPU lease: the inference worker's model goes before a GPU job loads its own."""
        sock = self.inference_socket
        if sock is None or not sock.exists():
            return
        from finetune_studio.supervisor.client import UnixHTTPConnection

        def _post() -> int:
            conn = UnixHTTPConnection(str(sock), 60.0)
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
        job = self._get(job_id)
        running = job.proc.returncode is None
        return {"id": job_id, "kind": job.spec.kind, "title": job.spec.title, "project_id": job.spec.project_id,
                "gpu": job.spec.gpu, "pid": job.proc.pid, "running": running,
                "uptime_s": round(time.time() - job.started_wall, 1) if running else None,
                "stop_requested": job.stop_requested,
                "exit": None if running else store.read_exit(job_id)}

    def logs(self, job_id: str, lines: int = 100) -> list[str]:
        return list(self._get(job_id).logs)[-lines:]

    def snapshot(self) -> list[dict[str, Any]]:
        return [self.describe(j) for j in self._jobs]
