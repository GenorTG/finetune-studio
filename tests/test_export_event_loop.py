"""POST /export must not block the event loop (live bug 2026-10-07).

During a GGUF export the whole server froze: the handler ran the merge +
convert + quantize (blocking ``subprocess.run``) inline on the loop. Here a
fake convert script sleeps for seconds as a REAL child process while other
requests are served through the same ASGI app on the same loop.
"""
from __future__ import annotations

import asyncio
import time
from pathlib import Path

import httpx
import pytest
from fastapi import FastAPI

from finetune_studio import db
from finetune_studio.webui import export_jobs
from finetune_studio.webui.routes import exports, projects
from tests._fake_llama import install_fake_llama, make_merged_run

SLOW_SECONDS = 3.0


def _app() -> FastAPI:
    app = FastAPI()
    app.include_router(projects.router, prefix="/api/projects")
    app.include_router(exports.router, prefix="/api")
    return app


@pytest.mark.parametrize("fmt_body", [
    {"format": "gguf", "quants": ["f16"]},      # multi-format path
    {"format": "gguf", "quant": "Q8_0"},         # single-quant path
])
def test_loop_keeps_serving_while_an_export_subprocess_runs(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, fmt_body: dict,
) -> None:
    install_fake_llama(tmp_path / "llama", monkeypatch)
    monkeypatch.setenv("FAKE_CONVERT_SECONDS", str(SLOW_SECONDS))
    pid = db.create_project(name="Loop", description="")["id"]
    run = make_merged_run(db, pid, tmp_path)

    async def drive() -> tuple[float, float, int, str]:
        transport = httpx.ASGITransport(app=_app())
        async with httpx.AsyncClient(transport=transport, base_url="http://t") as c:
            t0 = time.monotonic()
            r = await c.post(f"/api/projects/{pid}/runs/{run['id']}/export", json=fmt_body)
            post_s = time.monotonic() - t0
            assert r.status_code == 202, r.text
            eid = r.json()["export_id"]
            assert r.json()["status"] == "queued"

            worst, probes, running_seen = 0.0, 0, False
            while time.monotonic() - t0 < SLOW_SECONDS - 0.5:
                t1 = time.monotonic()
                lst = await c.get("/api/projects")
                row = await c.get(f"/api/projects/{pid}/exports/{eid}")
                worst = max(worst, time.monotonic() - t1)
                assert lst.status_code == 200 and row.status_code == 200
                running_seen |= row.json()["status"] in ("queued", "running")
                probes += 1
                await asyncio.sleep(0.05)
            assert running_seen, "the export finished before the probes ran; test is vacuous"
            for _ in range(200):
                final = (await c.get(f"/api/projects/{pid}/exports/{eid}")).json()
                if final["status"] in ("done", "failed", "cancelled"):
                    break
                await asyncio.sleep(0.1)
            return post_s, worst, probes, final["status"]

    post_s, worst, probes, status = asyncio.run(drive())
    assert post_s < 1.0, f"POST /export held the request for {post_s:.2f}s"
    assert probes >= 10
    assert worst < 1.0, f"event loop was blocked: slowest probe {worst:.2f}s"
    assert status == "done"
    deadline = time.monotonic() + 10
    while export_jobs.active_ids() and time.monotonic() < deadline:
        time.sleep(0.05)


def test_probe_detects_a_handler_that_blocks_the_loop() -> None:
    """Control: the same probe must flag the old inline-subprocess behaviour."""
    import subprocess
    import sys

    app = FastAPI()

    @app.post("/old-export")
    async def old_export() -> dict:
        subprocess.run([sys.executable, "-c", "import time; time.sleep(1.5)"], check=True)  # noqa: ASYNC221
        return {"ok": True}

    @app.get("/ping")
    async def ping() -> dict:
        return {"pong": True}

    async def drive() -> float:
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://t") as c:
            t0 = time.monotonic()
            slow = asyncio.create_task(c.post("/old-export"))
            await asyncio.sleep(0)      # the slow handler starts here and blocks the loop
            await c.get("/ping")
            worst = time.monotonic() - t0
            await slow
            return worst

    assert asyncio.run(drive()) > 1.0
