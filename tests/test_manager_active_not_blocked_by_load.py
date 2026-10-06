"""ModelManager.active() must answer while a model load holds the manager lock.

Found in the live walkthrough: starting helper mining froze every page/status poll for ~56 s, because async
route handlers called active(), which waited on the lock load() holds for the whole load.
"""
from __future__ import annotations

import threading
import time

from finetune_studio.models import manager as mgr_mod


class _Provider:
    _loaded_at = 0.0

    def describe(self) -> dict:
        return {"id": "p", "loaded": True}


def test_active_returns_immediately_while_load_holds_the_lock() -> None:
    m = mgr_mod.ModelManager.__new__(mgr_mod.ModelManager)
    m._lock = threading.RLock()
    m._provider = _Provider()
    release = threading.Event()
    held = threading.Event()

    def long_load() -> None:
        with m._lock:
            held.set()
            release.wait(5)

    t = threading.Thread(target=long_load)
    t.start()
    assert held.wait(2)
    start = time.time()
    got: list = []
    probe = threading.Thread(target=lambda: got.append(m.active()))
    probe.start()
    probe.join(1.0)
    release.set()
    t.join()
    assert got and got[0]["id"] == "p", "active() blocked on the manager lock"
    assert time.time() - start < 1.5


def test_active_is_none_when_nothing_is_loaded() -> None:
    m = mgr_mod.ModelManager.__new__(mgr_mod.ModelManager)
    m._lock = threading.RLock()
    m._provider = None
    assert m.active() is None
