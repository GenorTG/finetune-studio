"""Start, stop and look at jobs from the web app or the CLI.

With a supervisor (systemd unit / ``fts up``) it owns the job process; without one (``fts webui``, ``run.sh``) the
runner is started in its own session, so it still outlives a restart of the web process. Either way the job's
files (``jobs.store``) are the shared truth: state, exit record, pid.

``FTS_JOBS_INPROCESS=1`` (the test suite, ``tests/conftest.py``) runs the same kind code through the same
``runner.execute`` on a thread of this process instead: test monkeypatches reach it and nothing outlives the test.
``tests/test_jobs.py`` turns it off to exercise the real process boundary.
"""
from __future__ import annotations

import os
import signal
import subprocess
import sys
import threading
import time
from pathlib import Path
from typing import Any

from finetune_studio.jobs import store
from finetune_studio.supervisor.client import (
    SupervisorClient,
    SupervisorError,
    SupervisorUnavailable,
)


class JobRejected(RuntimeError):
    def __init__(self, status: int, message: str, code: str = "") -> None:
        super().__init__(message)
        self.status, self.code = status, code


# A start call answers once the job process is spawned; a GPU job's start first waits for the inference worker to
# unload (``supervisor.jobs.UNLOAD_TIMEOUT_S``, behind a model load that can take tens of seconds), so the call waits
# longer than that. A shorter wait answered 500 while the job started anyway (2026-11-13).
START_TIMEOUT_S = 120.0
# Without a supervisor a waiting GPU job (``JobSpec.queue_gpu``) is queued by this process: how often it checks.
LOCAL_QUEUE_POLL_S = 1.0


def _client() -> SupervisorClient | None:
    c = SupervisorClient(timeout=30.0)
    try:
        c.status()
    except SupervisorUnavailable:
        return None
    return c


def gpu_holder(*, exclude: str = "") -> dict[str, Any] | None:
    """The job holding the GPU lease (None = free): a running GPU job that has not released it. A job that only
    waits for the lease (``queued``) does not hold it."""
    return next((j for j in list_jobs(running_only=True)
                 if j["id"] != exclude and j.get("gpu") and not j.get("queued") and not store.gpu_released(j["id"])),
                None)


def holder_text(job: dict[str, Any]) -> str:
    return f"{job['kind']} job {job['id']} ({job.get('title') or 'untitled'})"


_HOLDER_LABELS = {"brain_build": "a Smart Brain build is running", "training": "a training run is running",
                  "export": "an export is running", "pictures": "pictures are being read by the Vision reader"}


def gpu_busy_info() -> dict[str, Any] | None:
    """What holds the GPU lease, for a refusal the user reads (None = free): the job, its plain-words label and how
    long it has been running (from its process start; a job that waited in the queue counts from when it got the
    lease)."""
    job = gpu_holder()
    if job is None:
        return None
    started = None
    try:
        pid_file = store.job_dir(job["id"]) / "pid"
        started = pid_file.stat().st_mtime if pid_file.exists() else None
    except (OSError, ValueError):
        started = None
    started = started or float(job.get("created_at") or time.time())
    return {"job_id": job["id"], "kind": job["kind"], "title": job.get("title") or "",
            "label": holder_label(holder_text(job)), "running_s": max(0, int(time.time() - started))}


def duration_text(seconds: float) -> str:
    """``45 s`` / ``12 min`` / ``1 h 05 min``."""
    s = int(seconds)
    if s < 60:
        return f"{s} s"
    if s < 3600:
        return f"{s // 60} min"
    return f"{s // 3600} h {s % 3600 // 60:02d} min"


def gpu_busy_message(info: dict[str, Any], what: str = "A chat model cannot load") -> str:
    """The refusal text: what holds the GPU, for how long, and how to free it."""
    return (f"{what}: the GPU is busy — {info['label']} (for {duration_text(info['running_s'])}, job "
            f"{info['job_id']}). Load it when that ends, or stop that job on its page (or `fts jobs stop "
            f"{info['job_id']}`).")


def holder_label(holder: str) -> str:
    """Plain words for what holds the GPU, from ``holder_text`` / a queue record ("a Smart Brain build is
    running"); the record's own words when it is no job ("the GPU to be freed ...")."""
    kind = holder.split(" job ", 1)[0].strip() if " job " in holder else ""
    if kind in _HOLDER_LABELS:
        return _HOLDER_LABELS[kind]
    if kind:
        return f"a {kind.replace('_', ' ')} job is using it"
    return holder or "another job is using it"


def start(spec: store.JobSpec) -> dict[str, Any]:
    """Write the spec and run it. Raises JobRejected (409 when the GPU is busy). A ``queue_gpu`` job is not
    refused then: it waits for the lease (``describe`` says ``queued`` and behind what)."""
    if spec.gpu:
        busy = gpu_holder()
        if busy and not spec.queue_gpu:
            raise JobRejected(409, f"the GPU is busy with {holder_text(busy)}; wait for it or stop it", "gpu_busy")
    if not spec.db_path:
        from finetune_studio.db import connection
        db = connection.settings.db_path
        spec.db_path = db if db == ":memory:" else os.path.abspath(db)
    store.write_spec(spec)
    client = None if _inprocess() else _client()
    if spec.gpu and spec.queue_gpu and client is None:
        busy = gpu_holder(exclude=spec.id)
        if busy:
            return _queue_locally(spec, holder_text(busy))
    return _launch(spec, client)


def _launch(spec: store.JobSpec, client: SupervisorClient | None) -> dict[str, Any]:
    if _inprocess():
        return _start_inprocess(spec)
    if client is not None:
        try:
            return SupervisorClient(client.path, timeout=START_TIMEOUT_S).start_job(spec.id)
        except SupervisorError as e:
            store.write_exit(spec.id, {"status": "failed", "error": str(e), "code": None})
            raise JobRejected(e.status, str(e), "gpu_busy" if e.status == 409 else "") from e
        except SupervisorUnavailable as e:
            # No answer in time: the supervisor may still have started it. Running -> the caller has its job;
            # otherwise a real 503 with the reason (never a 500, never a job left to start behind a "failed").
            try:
                job = describe(spec.id)
            except (FileNotFoundError, ValueError):
                job = None
            if job is not None and (job["running"] or store.read_pid(spec.id)):
                return job
            store.write_exit(spec.id, {"status": "failed", "error": f"not started: {e}", "code": None})
            raise JobRejected(503, f"the supervisor did not answer the job start: {e}", "supervisor_unavailable") from e
    # No supervisor: a detached runner (new session: neither a web restart nor Ctrl+C in its terminal kills it).
    log_path = store.job_dir(spec.id) / "output.log"
    with log_path.open("ab") as out:
        proc = subprocess.Popen(
            [sys.executable, "-m", "finetune_studio.jobs.runner", "--id", spec.id],
            cwd=os.getcwd(),          # the main DB path is cwd-relative: the job must see the web app's DB
            stdin=subprocess.DEVNULL, stdout=out, stderr=subprocess.STDOUT, start_new_session=True,
        )
    store.write_pid(spec.id, proc.pid)
    return describe(spec.id)


# ── waiting for the GPU without a supervisor (``fts webui``, the test suite) ───
_LOCAL_QUEUE: list[str] = []
_LOCAL_QUEUE_LOCK = threading.Lock()


def _queue_locally(spec: store.JobSpec, holder: str) -> dict[str, Any]:
    """This process holds the queue (it dies with it: ``describe`` then reads the job as crashed)."""
    store.write_queued(spec.id, {"waiting_for": holder, "owner_pid": os.getpid()})
    with _LOCAL_QUEUE_LOCK:
        _LOCAL_QUEUE.append(spec.id)
    threading.Thread(target=_local_wait, args=(spec.id,), daemon=True, name=f"gpu-wait-{spec.id[:20]}").start()
    return describe(spec.id)


def _local_wait(job_id: str) -> None:
    while True:
        time.sleep(LOCAL_QUEUE_POLL_S)
        if store.read_exit(job_id) is not None:          # stopped while it waited
            break
        with _LOCAL_QUEUE_LOCK:
            first = bool(_LOCAL_QUEUE) and _LOCAL_QUEUE[0] == job_id
        busy = gpu_holder(exclude=job_id)
        if busy is not None or not first:
            rec = store.read_queued(job_id) or {}
            if busy is not None and rec.get("waiting_for") != holder_text(busy):
                store.write_queued(job_id, {**rec, "waiting_for": holder_text(busy)})
            continue
        try:
            _launch(store.read_spec(job_id), None if _inprocess() else _client())
        except Exception as e:  # noqa: BLE001 - recorded on the job, never a dead thread with nothing to show
            store.write_exit(job_id, {"status": "failed", "error": f"not started after waiting for the GPU: {e}",
                                      "code": None})
        break
    with _LOCAL_QUEUE_LOCK:
        if job_id in _LOCAL_QUEUE:
            _LOCAL_QUEUE.remove(job_id)
    store.clear_queued(job_id)


# ── in-process executor (tests) ─────────────────────────────────────────
_INPROC: dict[str, tuple[threading.Thread, Any]] = {}
_INPROC_LOCK = threading.Lock()


def _inprocess() -> bool:
    return os.environ.get("FTS_JOBS_INPROCESS") == "1"


def _start_inprocess(spec: store.JobSpec) -> dict[str, Any]:
    from finetune_studio.jobs import runner
    ctx = runner.JobContext(spec=spec)
    t = threading.Thread(target=runner.execute, args=(ctx,), daemon=True, name=f"job-{spec.id[:20]}")
    with _INPROC_LOCK:
        _INPROC[spec.id] = (t, ctx)
    t.start()
    return describe(spec.id)


def _inproc_alive(job_id: str) -> bool | None:
    """None = not an in-process job."""
    with _INPROC_LOCK:
        entry = _INPROC.get(job_id)
    return None if entry is None else entry[0].is_alive()


def stop(job_id: str) -> dict[str, Any]:
    with _INPROC_LOCK:
        entry = _INPROC.get(job_id)
    if entry is not None:
        entry[1].stop.set()
        return describe(job_id)
    with _LOCAL_QUEUE_LOCK:
        local = job_id in _LOCAL_QUEUE
    if local:
        store.write_exit(job_id, {"status": "stopped", "detail": "stopped while waiting for the GPU", "code": None})
        store.clear_queued(job_id)
        return describe(job_id)
    client = _client()
    if client is not None:
        try:
            return client.stop_job(job_id)
        except SupervisorError as e:
            if e.status != 404:
                raise JobRejected(e.status, str(e)) from e
    pid = store.read_pid(job_id)
    if pid and store.pid_alive(job_id):
        try:
            if hasattr(os, "killpg"):
                os.killpg(pid, signal.SIGTERM)   # the runner sets ctx.stop; the kind ends cooperatively
            else:
                os.kill(pid, signal.SIGTERM)     # Windows: TerminateProcess (no cooperative stop there)
        except (ProcessLookupError, PermissionError):
            pass
    return describe(job_id)


def describe(job_id: str) -> dict[str, Any]:
    spec = store.read_spec(job_id)
    inproc = _inproc_alive(job_id)
    queued = _waiting(job_id) if inproc is None and not store.read_pid(job_id) else None
    if queued is not None:
        return {"id": job_id, "kind": spec.kind, "title": spec.title, "project_id": spec.project_id,
                "gpu": spec.gpu, "pid": None, "running": True, "exit": None, "queued": True,
                "waiting_for": str(queued.get("waiting_for") or ""), "waiting_since": queued.get("since"),
                "state": {"phase": "waiting_for_gpu", "waiting_for": str(queued.get("waiting_for") or "")},
                "created_at": spec.created_at}
    alive = inproc if inproc is not None else (store.pid_alive(job_id) or _starting(job_id))
    ex = None if alive else store.read_exit(job_id)
    if not alive and ex is None and inproc is None and store.read_pid(job_id):
        # Died without a record (killed, OOM-killed, power loss): say so instead of 'running' forever.
        ex = {"status": "crashed", "detail": "the job process is gone without an exit record"}
        store.write_exit(job_id, ex)
    return {"id": job_id, "kind": spec.kind, "title": spec.title, "project_id": spec.project_id, "gpu": spec.gpu,
            "pid": store.read_pid(job_id), "running": alive, "exit": ex, "queued": False,
            "state": store.read_state(job_id), "created_at": spec.created_at}


def _waiting(job_id: str) -> dict[str, Any] | None:
    """The queue record of a job that waits for the GPU lease, None when it does not wait. A record whose owner
    (the supervisor, or this web process without one) is gone is a job nobody will start: recorded as crashed."""
    rec = store.read_queued(job_id)
    if rec is None or store.read_exit(job_id) is not None:
        return None
    owner = int(rec.get("owner_pid") or 0)
    if owner and not _pid_exists(owner):
        store.clear_queued(job_id)
        store.write_exit(job_id, {"status": "crashed", "detail": "the process that queued it for the GPU is gone",
                                  "code": None})
        return None
    return rec


def _pid_exists(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    except OSError:
        return False
    return True


STARTING_GRACE_S = 15.0


def _starting(job_id: str) -> bool:
    """Spawned moments ago and not exec'd into the runner yet: not a crash."""
    pid_file = store.job_dir(job_id) / "pid"
    return (pid_file.exists() and store.read_exit(job_id) is None
            and time.time() - pid_file.stat().st_mtime < STARTING_GRACE_S
            and Path(f"/proc/{store.read_pid(job_id)}").exists())


def list_jobs(kind: str | None = None, *, running_only: bool = False) -> list[dict[str, Any]]:
    out = []
    for job_id in store.list_ids():
        try:
            d = describe(job_id)
        except (FileNotFoundError, ValueError, TypeError):
            continue
        if (kind is None or d["kind"] == kind) and (not running_only or d["running"]):
            out.append(d)
    return out


def wait(job_id: str, timeout: float = 30.0, poll: float = 0.1) -> dict[str, Any]:
    """Block until the job ends (tests, CLI). Returns its description."""
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        d = describe(job_id)
        if not d["running"] and d["exit"] is not None:
            return d
        time.sleep(poll)
    raise TimeoutError(f"job {job_id} still running after {timeout} s")
