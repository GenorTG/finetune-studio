"""Mining runs in the mining WORKER, a job process of its own (supervisor P3 for mining, 2026-10-15): a web restart no
longer ends it. Real processes here (the rest of the suite runs jobs on a thread, ``jobs.launch``): the helper is an
OpenAI-compatible stub served by this test, the worker reads the API helper row from the same models DB (``FTS_DB``) and
the web app's main DB (the spec carries its absolute path). A cancel from the web side reaches the worker through the
row; stopping the worker interrupts its files; the startup recovery leaves files whose worker is alive alone."""
from __future__ import annotations

import json
import os
import threading
import time
import uuid
from collections.abc import Iterator
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any

import pytest
from fastapi.testclient import TestClient

from finetune_studio import db
from finetune_studio.data.mining import scheduler, worker
from finetune_studio.jobs import launch, store
from finetune_studio.webui.app import app

DOC = """Leave policy extract

Annual leave is 21 working days for employees with 0-2 years of service at Korvane.

Lane | Rate EUR
RC-01 | 38.50
RC-02 | 41.75
"""
TERMINAL = ("done", "gaps", "failed", "cancelled", "interrupted")


class Stub:
    """An OpenAI-compatible chat endpoint: answers every call with an empty JSON list after ``delay`` seconds."""

    def __init__(self) -> None:
        self.calls = 0
        self.delay = 0.0
        stub = self

        class Handler(BaseHTTPRequestHandler):
            def do_POST(self) -> None:
                self.rfile.read(int(self.headers.get("Content-Length") or 0))
                stub.calls += 1
                time.sleep(stub.delay)
                body = json.dumps({"choices": [{"message": {"role": "assistant", "content": "[]"}}]}).encode()
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)

            def log_message(self, *_: Any) -> None:
                pass

        self.server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self.url = f"http://127.0.0.1:{self.server.server_address[1]}/v1"
        threading.Thread(target=self.server.serve_forever, daemon=True).start()


@pytest.fixture
def real_mining(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Iterator[tuple[TestClient, Stub]]:
    import finetune_studio.models.manager as mgr_mod
    from finetune_studio.models.helper import set_helper_provider_id

    monkeypatch.delenv("FTS_JOBS_INPROCESS", raising=False)
    monkeypatch.chdir(tmp_path)                                   # a stray cwd-relative DB would land here
    models_db = tmp_path / "models.db"
    monkeypatch.setenv("FTS_DB", str(models_db))                  # the worker process reads the same provider rows
    monkeypatch.setattr(mgr_mod, "_DB_PATH", models_db)
    monkeypatch.setattr(mgr_mod, "_manager", None)
    monkeypatch.setenv("NO_PROXY", "127.0.0.1,localhost")
    stub = Stub()
    mgr_mod.get_manager().upsert_provider(id="api-helper", name="Helper · API", kind="openai_compat", model_id="stub",
                                          base_url=stub.url, api_key="test-key", extra={"concurrency": 2, "retries": 0})
    set_helper_provider_id("api-helper")
    yield TestClient(app), stub
    for j in launch.list_jobs(scheduler.KIND, running_only=True):
        launch.stop(j["id"])
        launch.wait(j["id"], timeout=60)
    stub.server.shutdown()


def _file(client: TestClient) -> tuple[str, str]:
    pid = client.post("/api/projects", json={"name": f"mjp-{uuid.uuid4().hex[:6]}", "base_model": "x"}).json()["id"]
    r = client.post(f"/api/projects/{pid}/files/upload", files={"files": ("leave.txt", DOC.encode())})
    assert r.status_code == 200, r.text
    return pid, r.json()["report"][0]["source_id"]


def _submit(client: TestClient, pid: str, sid: str) -> dict:
    r = client.post(f"/api/projects/{pid}/mining/jobs", json={"source_ids": [sid], "kind": "mine"})
    assert r.status_code == 200, r.text
    return r.json()["jobs"][0]


def _wait_row(jid: str, until, timeout: float = 90) -> dict:
    deadline = time.monotonic() + timeout
    row: dict = {}
    while time.monotonic() < deadline:
        row = db.mining.get_job(jid) or {}
        if until(row):
            return row
        time.sleep(0.1)
    raise AssertionError(f"timed out: {row}; workers: {launch.list_jobs(scheduler.KIND)}")


def test_mining_runs_in_the_worker_process_against_the_web_apps_db(real_mining, tmp_path) -> None:
    client, stub = real_mining
    pid, sid = _file(client)
    job = _submit(client, pid, sid)
    assert job["helper"]["target"] == {"kind": "api", "provider_id": "api-helper"}
    row = _wait_row(job["id"], lambda r: r.get("status") in TERMINAL)
    assert row["status"] in ("done", "gaps"), row
    assert stub.calls > 0 and db.mining.list_rounds(pid, sid)
    d = launch.wait(row["runner"], timeout=60)                     # the worker exits once the queue is empty
    assert d["kind"] == "mining" and d["pid"] != os.getpid() and d["exit"]["status"] == "done"
    assert store.read_spec(row["runner"]).db_path == os.path.abspath(db.connection.settings.db_path)
    assert not (tmp_path / "data").exists(), "the worker used a cwd-relative DB instead of the web app's"


def test_cancel_from_the_web_side_reaches_the_worker(real_mining) -> None:
    client, stub = real_mining
    stub.delay = 1.0
    pid, sid = _file(client)
    job = _submit(client, pid, sid)
    row = _wait_row(job["id"], lambda r: r.get("status") == "running" and stub.calls > 0)
    assert row["runner"] in scheduler.running_mining_job_ids()
    r = client.post(f"/api/projects/{pid}/mining/jobs/{job['id']}/cancel")
    assert r.status_code == 200, r.text
    assert _wait_row(job["id"], lambda r: r.get("status") in TERMINAL, timeout=30)["status"] == "cancelled"
    assert client.post(f"/api/projects/{pid}/mining/jobs/{job['id']}/cancel").status_code == 409


def test_a_web_restart_leaves_a_running_file_alone_and_stopping_the_worker_interrupts_it(real_mining) -> None:
    client, stub = real_mining
    stub.delay = 1.0
    pid, sid = _file(client)
    job = _submit(client, pid, sid)
    row = _wait_row(job["id"], lambda r: r.get("status") == "running" and stub.calls > 0)
    out = scheduler.recover_after_restart()                       # what the web app runs at startup
    assert out["interrupted"] == 0 and db.mining.get_job(job["id"])["status"] == "running"
    launch.stop(row["runner"])
    assert launch.wait(row["runner"], timeout=60)["exit"]["status"] == "stopped"
    final = db.mining.get_job(job["id"])
    assert final["status"] == "interrupted" and final["error"] == worker.STOPPED_REASON, final


# ── in-process units (no worker process) ────────────────────────────────────────────────────────────────────────────

def test_a_file_whose_worker_is_gone_is_interrupted_and_queued_files_are_kept(monkeypatch: pytest.MonkeyPatch) -> None:
    client = TestClient(app)
    pid, sid = _file(client)
    lost = db.mining.create_job(pid, sid, "leave.txt", "mine")
    db.mining.update_job(lost["id"], status="running", runner="mining-20261015-000000-dead00")
    queued = db.mining.create_job(pid, sid, "leave.txt", "fill")
    started: list[str] = []
    monkeypatch.setattr(scheduler, "ensure_worker", lambda: started.append("w") or "mining-new")
    out = scheduler.recover_after_restart()
    assert out == {"interrupted": 1, "queued": 1, "worker": "mining-new"} and started == ["w"]
    assert db.mining.get_job(lost["id"])["status"] == "interrupted"
    assert db.mining.get_job(queued["id"])["status"] == "queued"


def test_a_worker_that_stopped_taking_jobs_gets_a_successor(monkeypatch: pytest.MonkeyPatch) -> None:
    alive = {"id": "mining-20261015-000000-aaaaaa", "kind": "mining", "running": True, "state": {"accepting": True}}
    monkeypatch.setattr(launch, "list_jobs", lambda kind=None, running_only=False: [alive])
    monkeypatch.setattr(launch, "start", lambda spec: pytest.fail("a worker that takes jobs is reused"))
    assert scheduler.ensure_worker() == alive["id"]
    alive["state"] = {"accepting": False}                       # it is exiting: a new file needs a new worker
    specs: list[store.JobSpec] = []
    monkeypatch.setattr(launch, "start", lambda spec: specs.append(spec) or {})
    new = scheduler.ensure_worker()
    assert new != alive["id"] and specs[0].kind == "mining" and specs[0].gpu is False


def test_a_local_helper_is_only_used_while_it_is_the_loaded_model_and_never_loaded(monkeypatch) -> None:
    from finetune_studio.data.mining import seat

    class Engine:
        model: Any = None
        model_path = ""

        def load(self, *a: Any, **k: Any) -> None:
            pytest.fail("mining must never load the local helper")

        def generate(self, messages: list, **gen: Any) -> str:
            return "ok"

    engine = Engine()
    monkeypatch.setattr("finetune_studio.inference.get_engine", lambda: engine)
    chat = seat.chat_for({"kind": "local", "model_path": "/m/helper.gguf"})
    with pytest.raises(RuntimeError, match="no longer loaded.*nothing is loaded"):
        chat([{"role": "user", "content": "x"}])
    engine.model, engine.model_path = object(), "/m/other.gguf"
    with pytest.raises(RuntimeError, match="now other.gguf"):
        chat([{"role": "user", "content": "x"}])
    engine.model_path = "/m/helper.gguf"
    assert chat([{"role": "user", "content": "x"}]) == "ok"
