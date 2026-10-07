"""An API helper serves requests in parallel (mining chunks concurrently); a local GGUF stays serialised."""
from __future__ import annotations

import threading
import time
from types import SimpleNamespace
from typing import Any

import pytest

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


def test_runner_uses_parallel_workers_only_for_an_api_helper(monkeypatch: pytest.MonkeyPatch) -> None:
    from finetune_studio.data.prep.runner import DataPrepRunner

    runner = DataPrepRunner("pid", b"x", "f.txt")
    for kind, expected in (("openai_compat", 6), ("local_gguf", 1)):
        monkeypatch.setattr(mgr_mod, "get_manager", lambda k=kind: SimpleNamespace(active=lambda: {"kind": k}))
        assert runner._api_workers() == expected
    monkeypatch.setenv("FTS_API_CONCURRENCY", "3")
    monkeypatch.setattr(mgr_mod, "get_manager", lambda: SimpleNamespace(active=lambda: {"kind": "openai_compat"}))
    assert runner._api_workers() == 3
