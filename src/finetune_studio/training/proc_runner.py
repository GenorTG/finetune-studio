"""Run a child process the way a long export job needs it.

``subprocess.run(..., timeout=)`` kills only the direct child on timeout, so a
``convert_hf_to_gguf.py`` that forked helpers (or a shell wrapper around
``llama-quantize``) leaves orphans holding gigabytes of RAM and the GPU. This
runner starts the child in its own session / process group, drains stdout and
stderr into a bounded tail (a multi-GB convert must not buffer its log), polls
for cancellation and a deadline, and on timeout or cancel kills the **whole
group** (SIGTERM, grace, SIGKILL) before it returns. The child is always dead
when :func:`run_group_subprocess` returns or raises.
"""

from __future__ import annotations

import contextlib
import os
import signal
import subprocess
import threading
import time
from collections import deque
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import IO

TAIL_CHARS = 4000
POLL_SECONDS = 0.25
TERM_GRACE_SECONDS = 5.0


class SubprocessTimeout(RuntimeError):
    """The child outlived its deadline and its process group was killed."""


class SubprocessCancelled(RuntimeError):
    """``cancelled()`` turned true; the child's process group was killed."""


class SubprocessFailed(RuntimeError):
    """Non-zero exit. ``tail`` is the end of stderr (stdout when stderr is empty)."""

    def __init__(self, cmd: str, returncode: int, tail: str) -> None:
        super().__init__(f"{cmd} failed (rc={returncode}): {tail}")
        self.returncode = returncode
        self.tail = tail


@dataclass(frozen=True)
class ProcResult:
    returncode: int
    stdout_tail: str
    stderr_tail: str
    seconds: float


class _Tail:
    """Thread-safe last-N-characters buffer fed by a pipe reader thread."""

    def __init__(self, limit: int) -> None:
        self._limit = limit
        self._chunks: deque[str] = deque()
        self._size = 0

    def add(self, text: str) -> None:
        self._chunks.append(text)
        self._size += len(text)
        while self._size - len(self._chunks[0]) >= self._limit and len(self._chunks) > 1:
            self._size -= len(self._chunks.popleft())

    def text(self) -> str:
        return "".join(self._chunks)[-self._limit:]


def _drain(stream: IO[str], tail: _Tail) -> None:
    try:
        for line in iter(stream.readline, ""):
            tail.add(line)
    except (OSError, ValueError):
        pass
    finally:
        with contextlib.suppress(OSError):
            stream.close()


def kill_process_group(proc: subprocess.Popen) -> None:
    """SIGTERM the child's group, wait a grace period, then SIGKILL it.

    The child was started with ``start_new_session`` so its pgid is its pid;
    killing by that id also reaches grandchildren after the leader is reaped.
    """
    for sig, grace in ((signal.SIGTERM, TERM_GRACE_SECONDS), (signal.SIGKILL, 2.0)):
        try:
            if hasattr(os, "killpg"):
                os.killpg(proc.pid, sig)
            elif sig == signal.SIGTERM:
                proc.terminate()
            else:
                proc.kill()
        except (ProcessLookupError, PermissionError):
            break
        deadline = time.monotonic() + grace
        while time.monotonic() < deadline:
            if proc.poll() is not None and _group_gone(proc.pid):
                return
            time.sleep(0.05)
    with contextlib.suppress(Exception):
        proc.wait(timeout=2)


def _group_gone(pgid: int) -> bool:
    """True when no process of group ``pgid`` is left (POSIX); else assume so."""
    if not hasattr(os, "killpg"):
        return True
    try:
        os.killpg(pgid, 0)
    except ProcessLookupError:
        return True
    except PermissionError:
        return False
    return False


def run_group_subprocess(
    cmd: Sequence[str],
    *,
    timeout: float,
    cancelled: Callable[[], bool] | None = None,
    on_tick: Callable[[], None] | None = None,
    on_start: Callable[[subprocess.Popen], None] | None = None,
    cwd: str | None = None,
    env: dict[str, str] | None = None,
) -> ProcResult:
    """Run ``cmd`` to completion; raise on failure, timeout or cancel.

    ``on_tick`` fires about every ``POLL_SECONDS`` while the child runs (the
    export job uses it for its heartbeat). ``on_start`` receives the ``Popen``
    right after spawn so a canceller can find it.
    """
    argv = [str(c) for c in cmd]
    name = os.path.basename(argv[0])
    popen_kw: dict = {}
    if hasattr(os, "killpg"):
        popen_kw["start_new_session"] = True
    elif hasattr(subprocess, "CREATE_NEW_PROCESS_GROUP"):
        popen_kw["creationflags"] = subprocess.CREATE_NEW_PROCESS_GROUP
    started = time.monotonic()
    proc = subprocess.Popen(
        argv, stdin=subprocess.DEVNULL, stdout=subprocess.PIPE,
        stderr=subprocess.PIPE, text=True, errors="replace",
        cwd=cwd, env=env, **popen_kw,
    )
    out_tail, err_tail = _Tail(TAIL_CHARS), _Tail(TAIL_CHARS)
    readers = [
        threading.Thread(target=_drain, args=(proc.stdout, out_tail), daemon=True),
        threading.Thread(target=_drain, args=(proc.stderr, err_tail), daemon=True),
    ]
    for t in readers:
        t.start()
    try:
        if on_start is not None:
            on_start(proc)
        while True:
            try:
                proc.wait(timeout=POLL_SECONDS)
                break
            except subprocess.TimeoutExpired:
                pass
            if on_tick is not None:
                on_tick()
            if cancelled is not None and cancelled():
                kill_process_group(proc)
                raise SubprocessCancelled(f"{name} cancelled")
            if time.monotonic() - started > timeout:
                kill_process_group(proc)
                raise SubprocessTimeout(
                    f"{name} timed out after {int(timeout)}s and was killed"
                    + _tail_suffix(err_tail, out_tail)
                )
    finally:
        # Whatever happened (incl. KeyboardInterrupt / on_tick raising), leave
        # no child behind.
        if proc.poll() is None or not _group_gone(proc.pid):
            kill_process_group(proc)
        for t in readers:
            t.join(timeout=2)
    result = ProcResult(
        proc.returncode, out_tail.text(), err_tail.text(), time.monotonic() - started,
    )
    if proc.returncode != 0:
        tail = (result.stderr_tail or result.stdout_tail)[-1000:]
        raise SubprocessFailed(name, proc.returncode, tail)
    return result


def _tail_suffix(err: _Tail, out: _Tail) -> str:
    tail = (err.text() or out.text())[-500:].strip()
    return f": {tail}" if tail else ""
