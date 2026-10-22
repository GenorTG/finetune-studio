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


def _client() -> SupervisorClient | None:
    c = SupervisorClient(timeout=30.0)
    try:
        c.status()
    except SupervisorUnavailable:
        return None
    return c


def start(spec: store.JobSpec) -> dict[str, Any]:
    """Write the spec and run it. Raises JobRejected (409 when the GPU is busy)."""
    if spec.gpu:
        busy = next((j for j in list_jobs() if j["running"] and j.get("gpu") and not store.gpu_released(j["id"])),
                    None)
        if busy:
            raise JobRejected(409, f"the GPU is busy with {busy['kind']} job {busy['id']} "
                                   f"({busy.get('title') or 'untitled'}); wait for it or stop it", "gpu_busy")
    if not spec.db_path:
        from finetune_studio.db import connection
        db = connection.settings.db_path
        spec.db_path = db if db == ":memory:" else os.path.abspath(db)
    store.write_spec(spec)
    if _inprocess():
        return _start_inprocess(spec)
    client = _client()
    if client is not None:
        try:
            return client.start_job(spec.id)
        except SupervisorError as e:
            store.write_exit(spec.id, {"status": "failed", "error": str(e), "code": None})
            raise JobRejected(e.status, str(e), "gpu_busy" if e.status == 409 else "") from e
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
    alive = inproc if inproc is not None else (store.pid_alive(job_id) or _starting(job_id))
    ex = None if alive else store.read_exit(job_id)
    if not alive and ex is None and inproc is None and store.read_pid(job_id):
        # Died without a record (killed, OOM-killed, power loss): say so instead of 'running' forever.
        ex = {"status": "crashed", "detail": "the job process is gone without an exit record"}
        store.write_exit(job_id, ex)
    return {"id": job_id, "kind": spec.kind, "title": spec.title, "project_id": spec.project_id, "gpu": spec.gpu,
            "pid": store.read_pid(job_id), "running": alive, "exit": ex, "state": store.read_state(job_id),
            "created_at": spec.created_at}


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
