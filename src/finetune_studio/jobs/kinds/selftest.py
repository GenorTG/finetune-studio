"""A job that needs nothing: counts to ``params.steps`` publishing progress, honours stop, can fail or crash on
request, and can leave a thread running when stopped (``leave_thread_s``) or raise ``Stopped`` (``raise_stopped``). Used by the tests (crash injection without a GPU) and by ``fts doctor --jobs``."""
from __future__ import annotations

import os
import signal
import threading
import time
from typing import Any

from finetune_studio.jobs.runner import JobContext, Stopped


def run(ctx: JobContext) -> dict[str, Any]:
    steps = int(ctx.params.get("steps", 5))
    delay = float(ctx.params.get("delay_s", 0.2))
    for i in range(1, steps + 1):
        if ctx.stop.is_set():
            ctx.publish({"status": "stopped", "step": i - 1, "total": steps}, force=True)
            if ctx.params.get("raise_stopped"):       # the other way a kind ends on a stop (the Smart Brain build)
                raise Stopped("stopped on request")
            if ctx.params.get("leave_thread_s"):      # like a stopped hub download: work left on a non-daemon thread
                threading.Thread(target=time.sleep, args=(float(ctx.params["leave_thread_s"]),), daemon=False).start()
            return {"status": "stopped", "step": i - 1}
        if ctx.params.get("fail_at") == i:
            raise RuntimeError(f"selftest failure at step {i}")
        if ctx.params.get("crash_at") == i:
            os.kill(os.getpid(), signal.SIGKILL)        # a hard death: no exit.json from the runner
        ctx.publish({"status": "running", "step": i, "total": steps}, force=True)
        time.sleep(delay)
    ctx.publish({"status": "done", "step": steps, "total": steps}, force=True)
    return {"status": "done", "steps": steps}
