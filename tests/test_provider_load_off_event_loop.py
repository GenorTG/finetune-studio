"""POST /providers/{pid}/load must not run the model load on the event loop.

Found in the live walkthrough: while the helper loaded (~50 s) every other request — page navigation, status
polls, SSE — stalled, because the async handler called get_manager().load() inline.
"""
from __future__ import annotations

import asyncio
import threading
import time

from fastapi import FastAPI
from fastapi.testclient import TestClient

from finetune_studio.models import manager as mgr_mod
from finetune_studio.webui.routes.data_prep import router


def test_load_runs_in_a_worker_thread_and_the_loop_stays_responsive(monkeypatch) -> None:
    seen: dict = {}

    class SlowMgr:
        def load(self, pid, extra=None):
            seen["thread"] = threading.current_thread().name
            time.sleep(1.0)
            return {"id": pid, "loaded": True}

    monkeypatch.setattr(mgr_mod, "get_manager", lambda: SlowMgr())
    monkeypatch.setattr("finetune_studio.models.helper.missing_gguf_for_provider", lambda pid: "")
    app = FastAPI()
    app.include_router(router)

    async def drive() -> float:
        import httpx

        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://t") as c:
            load = asyncio.create_task(c.post("/providers/p1/load", json={}))
            await asyncio.sleep(0.2)
            t0 = time.time()
            await asyncio.sleep(0)      # the loop must be free to run other tasks while the load sleeps
            ticks = 0
            while not load.done():
                await asyncio.sleep(0.05)
                ticks += 1
            assert (await load).status_code == 200
            assert ticks >= 5, "event loop was blocked by the model load"
            return time.time() - t0

    asyncio.run(drive())
    assert seen["thread"] != threading.main_thread().name


def test_the_route_still_reports_a_failed_load() -> None:
    from unittest.mock import patch

    class BadMgr:
        def load(self, pid, extra=None):
            raise RuntimeError("no memory")

    app = FastAPI()
    app.include_router(router)
    with patch.object(mgr_mod, "get_manager", lambda: BadMgr()), \
         patch("finetune_studio.models.helper.missing_gguf_for_provider", lambda pid: ""):
        r = TestClient(app).post("/providers/p1/load", json={})
    assert r.status_code == 400 and "no memory" in r.json()["error"]
