"""Progress + cancel context for one export job.

The export pipeline (``run_export`` -> ``gguf_convert`` / engine merge) is plain
blocking code that knows nothing about the web layer. The WebUI job runner
binds an :class:`ExportContext` to the worker thread; the pipeline reports
phases through :func:`report` and polls it for cancellation, and outside a job
(CLI, tests) every hook is a no-op. A ``ContextVar`` carries it so no function
signature in the pipeline has to change.
"""

from __future__ import annotations

import contextlib
import contextvars
import subprocess
import threading
import time
from collections.abc import Callable, Iterator

PHASE_QUEUED = "queued"
PHASE_PREPARING = "preparing"
PHASE_MERGING = "merging"
PHASE_CONVERTING = "converting"
PHASE_QUANTIZING = "quantizing"
PHASE_ABLITERATING = "abliterating"
PHASE_FINALIZING = "finalizing"

HEARTBEAT_EVERY = 2.0


class ExportCancelled(RuntimeError):
    """The user cancelled the export; raised at the next checkpoint."""


class ExportContext:
    """Per-job hooks: phase sink, heartbeat sink and a cancel flag."""

    def __init__(
        self,
        export_id: str,
        *,
        on_phase: Callable[[str, str], None],
        on_heartbeat: Callable[[], None],
    ) -> None:
        self.export_id = export_id
        self.cancel_event = threading.Event()
        self._on_phase = on_phase
        self._on_heartbeat = on_heartbeat
        self._last_beat = 0.0
        self._procs: list[subprocess.Popen] = []
        self._procs_lock = threading.Lock()

    def register_proc(self, proc: subprocess.Popen) -> None:
        with self._procs_lock:
            self._procs = [p for p in self._procs if p.poll() is None]
            self._procs.append(proc)

    def kill_children(self) -> None:
        """Kill the process group of every child still running (blocking)."""
        from finetune_studio.training.proc_runner import kill_process_group

        with self._procs_lock:
            procs, self._procs = self._procs, []
        for proc in procs:
            if proc.poll() is None:
                kill_process_group(proc)

    def cancelled(self) -> bool:
        return self.cancel_event.is_set()

    def check(self) -> None:
        if self.cancel_event.is_set():
            raise ExportCancelled("export cancelled")

    def phase(self, phase: str, detail: str = "") -> None:
        self.check()
        self._last_beat = time.monotonic()
        self._on_phase(phase, detail)

    def tick(self) -> None:
        """Throttled heartbeat; safe to call from a tight poll loop."""
        now = time.monotonic()
        if now - self._last_beat >= HEARTBEAT_EVERY:
            self._last_beat = now
            self._on_heartbeat()


_CTX: contextvars.ContextVar[ExportContext | None] = contextvars.ContextVar(
    "fts_export_ctx", default=None,
)


def current() -> ExportContext | None:
    return _CTX.get()


@contextlib.contextmanager
def bind(ctx: ExportContext) -> Iterator[ExportContext]:
    token = _CTX.set(ctx)
    try:
        yield ctx
    finally:
        _CTX.reset(token)


def report(phase: str, detail: str = "") -> None:
    """Record the pipeline phase for the running job (no-op outside a job).

    Doubles as a cancel checkpoint: raises :class:`ExportCancelled` when the
    user cancelled since the last call.
    """
    ctx = _CTX.get()
    if ctx is not None:
        ctx.phase(phase, detail)
