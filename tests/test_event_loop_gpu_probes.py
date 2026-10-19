"""GPU probes never run on the event loop.

Regression (2026-10-19): ``POST /api/inference/memory-estimate`` ran the VRAM probe inline; its first call after a restart
(CUDA init while the worker loaded a 12B) held the loop 33 s, the supervisor's health check timed out and the web server
was killed twice.
"""
from __future__ import annotations

import inspect
import threading
import time
from typing import Any

import pytest

from finetune_studio.webui.routes import pages, system


def test_probe_handlers_run_in_the_threadpool() -> None:
    for handler in (system.resources, system.accelerator, system.gpu_summary, system.gpu_text, pages.debug_info):
        assert not inspect.iscoroutinefunction(handler), handler.__name__


def test_memory_estimate_keeps_the_event_loop_free(client: Any, monkeypatch: pytest.MonkeyPatch) -> None:
    from finetune_studio.webui import app as webapp

    monkeypatch.setattr(webapp.inference_engine, "estimate_memory", lambda path, **kw: {"weights_gb": 1.0}, raising=False)
    monkeypatch.setattr(system, "_vram", lambda: time.sleep(1.5) or [{"used_gb": 1.0, "total_gb": 24.0}])
    done: dict[str, Any] = {}
    worker = threading.Thread(target=lambda: done.update(
        r=client.post("/api/inference/memory-estimate", json={"model_path": "/m/x.gguf"})))
    worker.start()
    time.sleep(0.3)
    t0 = time.perf_counter()
    assert client.get("/api/health").status_code == 200
    assert time.perf_counter() - t0 < 0.8, "the health check waited for the GPU probe"
    worker.join()
    assert done["r"].status_code == 200 and done["r"].json()["total_gpu_gb"] == 24.0
