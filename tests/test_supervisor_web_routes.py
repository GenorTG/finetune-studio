"""The web app's side of the supervisor contract: a cheap liveness route and the component table."""
from __future__ import annotations

import asyncio
import tempfile

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from finetune_studio.supervisor import control
from finetune_studio.supervisor.core import Supervisor
from finetune_studio.supervisor.events import EventLog
from finetune_studio.supervisor.paths import socket_path
from finetune_studio.supervisor.spec import ComponentSpec
from finetune_studio.webui.routes.system import router


@pytest.fixture
def client(monkeypatch):
    d = tempfile.mkdtemp(prefix="fts")
    monkeypatch.setenv("FTS_ROOT", d)
    app = FastAPI()
    app.include_router(router)
    yield TestClient(app)
    import shutil
    shutil.rmtree(d, ignore_errors=True)


def test_health_is_a_plain_200(client):
    r = client.get("/api/health")
    assert r.status_code == 200 and r.json() == {"status": "ok"}


def test_supervisor_route_reports_unmanaged_without_a_supervisor(client):
    assert client.get("/api/system/supervisor").json() == {"managed": False, "components": {}}


def test_supervisor_route_returns_the_component_table(client):
    import sys

    async def main():
        spec = ComponentSpec(name="web", argv=lambda: [sys.executable, "-c", "import time; time.sleep(30)"], ready_after=0.2)
        sup = Supervisor([spec], EventLog())
        server = await control.serve(sup, socket_path())
        runner = asyncio.create_task(sup.run())
        try:
            for _ in range(60):
                body = await asyncio.to_thread(lambda: client.get("/api/system/supervisor").json())
                if body.get("components", {}).get("web", {}).get("state") == "ready":
                    return body
                await asyncio.sleep(0.1)
            raise AssertionError("web component never became ready")
        finally:
            sup.shutdown()
            await runner
            await server.stop()

    body = asyncio.run(main())
    assert body["managed"] is True and body["components"]["web"]["pid"]
