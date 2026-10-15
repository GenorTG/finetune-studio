"""An API provider serves requests in parallel; a local GGUF stays serialised (the mining limiter: tests/test_mining.py)."""
from __future__ import annotations

import threading
import time
from typing import Any

from finetune_studio.models import manager as mgr_mod


class _Slow:
    def __init__(self, concurrent: bool) -> None:
        self.concurrent = concurrent
        self.active = 0
        self.peak = 0
        self._lock = threading.Lock()

    def chat(self, messages: list[dict], **gen: Any) -> str:
        with self._lock:
            self.active += 1
            self.peak = max(self.peak, self.active)
        time.sleep(0.15)
        with self._lock:
            self.active -= 1
        return "ok"


def _run_parallel(concurrent: bool) -> int:
    m = mgr_mod.ModelManager.__new__(mgr_mod.ModelManager)
    m._lock = threading.RLock()
    m._invoke_lock = threading.Lock()
    p = _Slow(concurrent)
    m._provider = p
    threads = [threading.Thread(target=m.chat, args=([{"role": "user", "content": "x"}],)) for _ in range(4)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    return p.peak


def test_remote_provider_calls_overlap_and_local_calls_do_not() -> None:
    assert _run_parallel(True) >= 3
    assert _run_parallel(False) == 1
