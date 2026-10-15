"""``python -m finetune_studio.jobs.runner --id <job id>``: run one job and leave its exit record.

Exit codes: 0 done, 1 failed, 2 bad spec/unknown kind, 3 stopped on request.
"""
from __future__ import annotations

import argparse
import importlib
import logging
import os
import signal
import sys
import threading
import time
import traceback
from dataclasses import dataclass, field
from typing import Any

from finetune_studio.jobs import store

# kind -> "module:function"; the function takes a JobContext and returns a result dict (or raises).
KINDS: dict[str, str] = {
    "selftest": "finetune_studio.jobs.kinds.selftest:run",
    "training": "finetune_studio.jobs.kinds.training:run",
    "export": "finetune_studio.jobs.kinds.export:run",
}

EXIT_DONE, EXIT_FAILED, EXIT_BAD_SPEC, EXIT_STOPPED = 0, 1, 2, 3
log = logging.getLogger("finetune_studio.jobs")


class Stopped(Exception):
    """A kind raises this (or returns after ``ctx.stop`` is set) to end as 'stopped'."""


@dataclass
class JobContext:
    spec: store.JobSpec
    stop: threading.Event = field(default_factory=threading.Event)
    _last_publish: float = 0.0

    @property
    def id(self) -> str:
        return self.spec.id

    @property
    def params(self) -> dict[str, Any]:
        return self.spec.params

    def path(self, name: str):
        return store.job_dir(self.id) / name

    def publish(self, state: dict[str, Any], *, force: bool = False, every_s: float = 0.5) -> None:
        """Live state for the pages. Throttled; ``force`` for status changes that must not be dropped."""
        now = time.monotonic()
        if force or now - self._last_publish >= every_s:
            self._last_publish = now
            store.publish_state(self.id, state)


def _resolve(kind: str):
    target = KINDS.get(kind)
    if target is None:
        raise KeyError(f"unknown job kind {kind!r}")
    module, func = target.split(":")
    return getattr(importlib.import_module(module), func)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m finetune_studio.jobs.runner")
    parser.add_argument("--id", required=True)
    args = parser.parse_args(argv)
    try:
        spec = store.read_spec(args.id)
        fn = _resolve(spec.kind)
    except (FileNotFoundError, KeyError, ValueError, TypeError) as e:
        print(f"[job] {e}", file=sys.stderr, flush=True)
        try:
            store.write_exit(args.id, {"status": "failed", "error": str(e), "code": EXIT_BAD_SPEC})
        except ValueError:
            pass
        return EXIT_BAD_SPEC

    from finetune_studio import logsetup
    logsetup.setup(f"job-{spec.kind}")
    if spec.db_path:
        from finetune_studio.config import settings
        settings.db_path = spec.db_path           # the web app's DB, not one relative to this cwd
    store.write_pid(spec.id, os.getpid())
    ctx = JobContext(spec=spec)

    def _on_term(signum: int, _frame: Any) -> None:
        if ctx.stop.is_set():
            return
        log.info("job %s: stop requested (signal %s)", spec.id, signal.Signals(signum).name)
        ctx.stop.set()

    signal.signal(signal.SIGTERM, _on_term)
    signal.signal(signal.SIGINT, _on_term)
    log.info("job %s (%s) started, pid %d", spec.id, spec.kind, os.getpid())
    return execute(ctx, fn)


def execute(ctx: JobContext, fn: Any = None) -> int:
    """Run the kind and leave its exit record. Shared by the job process and the in-process test executor."""
    spec = ctx.spec
    fn = fn or _resolve(spec.kind)
    started = time.time()
    try:
        result = fn(ctx) or {}
        status = str(result.get("status", "done"))
        code = {"done": EXIT_DONE, "stopped": EXIT_STOPPED}.get(status, EXIT_FAILED)
        store.write_exit(spec.id, {"status": status, "code": code, "result": result,
                                   "ran_s": round(time.time() - started, 1)}, overwrite=True)
        log.info("job %s ended: %s", spec.id, status)
        return code
    except Stopped:
        store.write_exit(spec.id, {"status": "stopped", "code": EXIT_STOPPED, "ran_s": round(time.time() - started, 1)},
                         overwrite=True)
        log.info("job %s stopped", spec.id)
        return EXIT_STOPPED
    except Exception as e:
        log.exception("job %s failed", spec.id)
        store.write_exit(spec.id, {"status": "failed", "code": EXIT_FAILED, "error": f"{type(e).__name__}: {e}",
                                   "traceback": traceback.format_exc()[-4000:],
                                   "ran_s": round(time.time() - started, 1)}, overwrite=True)
        return EXIT_FAILED

if __name__ == "__main__":
    sys.exit(main())
