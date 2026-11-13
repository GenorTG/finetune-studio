"""A slow job start never answers 500 and never starts a second GPU job (2026-11-13: ``POST /brain/build`` answered
500 ``SupervisorUnavailable`` while the supervisor waited for the inference worker to unload behind a model load;
the job started anyway and the retry started a second build of the same project beside it)."""
from __future__ import annotations

import asyncio
import os
import threading
import time

import pytest

from finetune_studio.jobs import launch, store
from finetune_studio.jobs.store import JobSpec
from finetune_studio.supervisor import client as sup_client
from finetune_studio.supervisor import jobs as sup_jobs


def _spec(gpu: bool = True) -> JobSpec:
    return JobSpec(id=store.new_id("selftest"), kind="selftest", params={"steps": 1}, gpu=gpu, stop_grace_s=5,
                   title="t")


def test_a_second_gpu_start_while_the_first_waits_for_the_unload_is_refused(tmp_path, monkeypatch) -> None:
    from finetune_studio.supervisor.events import EventLog

    monkeypatch.setenv("FTS_ROOT", str(tmp_path / "root"))
    spawned: list[str] = []

    class _Proc:
        returncode = 0
        pid = 4242
        stdout = None

        async def wait(self) -> int:
            return 0

    async def fake_exec(*args, **_kw):
        spawned.append(args[-1])
        return _Proc()

    async def scenario() -> None:
        table = sup_jobs.JobTable(EventLog(tmp_path / "events.jsonl"), cwd=os.getcwd())
        gate = asyncio.Event()

        async def slow_unload() -> None:          # the inference worker is busy loading a model
            await gate.wait()

        table._unload_inference_model = slow_unload          # type: ignore[method-assign]
        table._watch = lambda job: asyncio.sleep(0)          # type: ignore[method-assign]
        first, second = _spec(), _spec()
        store.write_spec(first)
        store.write_spec(second)
        t1 = asyncio.create_task(table.start(first.id))
        await asyncio.sleep(0.05)                            # first is inside the unload wait
        with pytest.raises(sup_jobs.JobBusy, match="starting"):
            await asyncio.wait_for(table.start(second.id), 5)
        with pytest.raises(sup_jobs.JobBusy):
            await asyncio.wait_for(table.start(first.id), 5)  # the same start asked twice
        gate.set()
        await t1
        assert spawned == [first.id]
        cpu = _spec(gpu=False)
        store.write_spec(cpu)
        await table.start(cpu.id)                            # a CPU job never waits for the GPU lease
        assert spawned == [first.id, cpu.id]

    monkeypatch.setattr(sup_jobs.asyncio, "create_subprocess_exec", fake_exec)
    asyncio.run(scenario())


def test_the_start_call_waits_longer_than_the_unload() -> None:
    assert launch.START_TIMEOUT_S > sup_jobs.UNLOAD_TIMEOUT_S


class _FakeClient:
    path = "/nowhere.sock"

    def __init__(self, *_a, **_kw) -> None:
        pass

    def status(self) -> dict:
        return {}


def test_a_start_that_timed_out_but_started_returns_the_job(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("FTS_ROOT", str(tmp_path / "root"))
    monkeypatch.delenv("FTS_JOBS_INPROCESS", raising=False)
    spec = _spec()

    class _Late(_FakeClient):
        def start_job(self, job_id: str) -> dict:
            store.write_pid(job_id, os.getpid())            # the supervisor did spawn it
            raise sup_client.SupervisorUnavailable("timed out")

    monkeypatch.setattr(launch, "_client", lambda: _Late())
    monkeypatch.setattr(launch, "SupervisorClient", _Late)
    out = launch.start(spec)
    assert out["id"] == spec.id and out["running"] is True
    assert store.read_exit(spec.id) is None


def test_a_start_that_got_no_answer_and_no_job_is_a_503(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("FTS_ROOT", str(tmp_path / "root"))
    monkeypatch.delenv("FTS_JOBS_INPROCESS", raising=False)
    spec = _spec()

    class _Dead(_FakeClient):
        def start_job(self, job_id: str) -> dict:
            raise sup_client.SupervisorUnavailable("timed out")

    monkeypatch.setattr(launch, "_client", lambda: _Dead())
    monkeypatch.setattr(launch, "SupervisorClient", _Dead)
    with pytest.raises(launch.JobRejected) as e:
        launch.start(spec)
    assert e.value.status == 503 and e.value.code == "supervisor_unavailable"
    assert store.read_exit(spec.id)["status"] == "failed"


def test_the_build_route_answers_503_not_500_when_the_supervisor_is_silent(client, monkeypatch) -> None:
    from finetune_studio import db
    from tests.knowledge_helpers import add_source

    pid = client.post("/api/projects", json={"name": "Silent supervisor"}).json()["id"]
    add_source(pid, "a.txt", "File A says the gate closes at 22:00.")

    def _silent(_spec):
        raise launch.JobRejected(503, "the supervisor did not answer the job start: timed out", "supervisor_unavailable")

    monkeypatch.setattr(launch, "start", _silent)
    r = client.post(f"/api/projects/{pid}/brain/build", json={})
    assert r.status_code == 503 and r.json()["code"] == "supervisor_unavailable", r.text
    assert db.brain_builds.latest_for_project(pid)["status"] == "error"


def test_two_build_posts_at_once_start_one_build(client, monkeypatch) -> None:
    from finetune_studio.webui import brain_build_jobs
    from tests.knowledge_helpers import add_source

    pid = client.post("/api/projects", json={"name": "Double click"}).json()["id"]
    add_source(pid, "a.txt", "File A says the gate closes at 22:00.")
    started: list[str] = []
    entered = threading.Event()

    def _slow_start(spec):                                   # the supervisor is waiting for the inference unload
        entered.set()
        time.sleep(0.4)
        started.append(spec.id)
        return {"id": spec.id, "running": True}

    monkeypatch.setattr(launch, "start", _slow_start)
    monkeypatch.setattr(brain_build_jobs, "_running",
                        lambda: [{"build_id": s.split("-", 1)[1], "project_id": pid} for s in started])
    codes: list[int] = []

    def post() -> None:
        codes.append(client.post(f"/api/projects/{pid}/brain/build", json={}).status_code)

    t = threading.Thread(target=post)
    t.start()
    assert entered.wait(10)
    post()
    t.join(20)
    assert sorted(codes) == [202, 409] and len(started) == 1
