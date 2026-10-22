"""Find or start the inference worker.

Order: a worker already answering on the socket (shared) -> the supervisor's ``inference`` component
(ask it to start, wait) -> a private worker spawned by THIS process (standalone web server or a CLI
command), stopped when this process exits.
"""
from __future__ import annotations

import atexit
import contextlib
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

from finetune_studio.supervisor.client import (
    SupervisorClient,
    SupervisorError,
    SupervisorUnavailable,
    UnixHTTPConnection,
)
from finetune_studio.supervisor.paths import inference_log_path, socket_path

START_TIMEOUT_S = 240.0  # importing torch + llama.cpp on a cold cache takes a while
COMPONENT = "inference"

_private: subprocess.Popen[bytes] | None = None
_private_sock: Path | None = None


def worker_alive(sock: Path, timeout: float = 2.0) -> bool:
    conn = UnixHTTPConnection(str(sock), timeout)
    try:
        conn.request("GET", "/v1/health")
        return conn.getresponse().status == 200
    except OSError:
        return False
    finally:
        conn.close()


def _supervisor_component() -> dict[str, Any] | None:
    if not socket_path().exists():
        return None
    try:
        return SupervisorClient(timeout=2.0).status()["components"].get(COMPONENT)
    except (SupervisorUnavailable, SupervisorError, KeyError):
        return None


def _wait_alive(sock: Path, deadline_s: float, on_fail: str) -> None:
    end = time.monotonic() + deadline_s
    while time.monotonic() < end:
        if worker_alive(sock):
            return
        if _private is not None and _private.poll() is not None:
            raise RuntimeError(f"the inference worker exited with code {_private.returncode} while starting; see {inference_log_path()}")
        time.sleep(0.25)
    raise RuntimeError(on_fail)


def ensure_worker(sock: Path) -> str:
    """Make sure a worker answers on ``sock``; returns ``shared`` | ``supervised`` | ``private``."""
    global _private, _private_sock
    if worker_alive(sock):
        return "shared"
    comp = _supervisor_component()
    if comp is not None:
        if comp["state"] in ("stopped", "failed"):
            SupervisorClient(timeout=3.0).act(COMPONENT, "start")
        _wait_alive(sock, START_TIMEOUT_S, "the supervisor's inference worker did not become ready; see `fts status` / `fts logs inference`")
        return "supervised"
    if _private is not None and _private.poll() is None:
        _wait_alive(sock, START_TIMEOUT_S, "the private inference worker did not become ready")
        return "private"
    _private = _spawn_private(sock)
    _private_sock = sock
    atexit.register(stop_private_worker)
    print(f"[inference] starting a private worker (pid {_private.pid}, log {inference_log_path()})", file=sys.stderr, flush=True)
    _wait_alive(sock, START_TIMEOUT_S, f"the private inference worker did not become ready; see {inference_log_path()}")
    return "private"


def _spawn_private(sock: Path, extra_args: tuple[str, ...] = ()) -> subprocess.Popen[bytes]:
    """Start a worker owned by this process (the test suite replaces this so no test can start a real one)."""
    from finetune_studio.supervisor.env import child_env

    sock.parent.mkdir(parents=True, exist_ok=True)
    with inference_log_path().open("ab") as log:
        return subprocess.Popen(
            [sys.executable, "-m", "finetune_studio.inference.worker", "--socket", str(sock),
             "--owner-pid", str(os.getpid()), *extra_args],
            stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT, env=child_env(), cwd=os.getcwd(),
        )


def stop_private_worker() -> None:
    """Stop the worker this process started (no-op for a shared or supervised one)."""
    global _private, _private_sock
    proc, _private = _private, None
    sock, _private_sock = _private_sock, None
    if proc is not None and proc.poll() is None:
        with contextlib.suppress(ProcessLookupError):
            proc.terminate()
        try:
            proc.wait(10)
        except subprocess.TimeoutExpired:
            with contextlib.suppress(ProcessLookupError):
                proc.kill()
    if sock is not None:  # the worker unlinks on a clean exit; a killed one cannot
        from finetune_studio.supervisor.client import port_file
        for leftover in (sock, port_file(sock)):
            with contextlib.suppress(FileNotFoundError):
                leftover.unlink()


def describe_worker_exit() -> str:
    """`` (killed by SIGSEGV)`` when the supervisor knows why the worker died, else ``""``."""
    comp = _supervisor_component()
    last = (comp or {}).get("last_exit") or {}
    if last.get("signal"):
        return f" (killed by {last['signal']})"
    if last.get("code") is not None:
        return f" (exit code {last['code']})"
    if _private is not None and _private.poll() is not None:
        rc = _private.returncode
        return f" (killed by signal {-rc})" if rc < 0 else f" (exit code {rc})"
    return ""


def worker_info(status: dict[str, Any]) -> dict[str, Any]:
    comp = _supervisor_component()
    if comp is not None:
        return {"transport": "supervised", "pid": comp.get("pid"), "state": comp.get("state"), "restarts": comp.get("restarts"),
                "last_exit": comp.get("last_exit"), "loaded": bool(status.get("loaded"))}
    if status.get("pid"):
        return {"transport": "private" if _private is not None else "shared", "pid": status.get("pid"), "state": "ready",
                "restarts": None, "last_exit": None, "loaded": bool(status.get("loaded"))}
    return {"transport": "none", "pid": None, "state": "stopped", "restarts": None, "last_exit": None, "loaded": False}
