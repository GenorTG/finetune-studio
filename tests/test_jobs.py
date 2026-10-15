"""Jobs as their own processes (supervisor P3, 2026-10-15): they outlive the web process, stop cleanly, and a hard
death is reported as such. Uses the ``selftest`` kind: no GPU, no model."""
from __future__ import annotations

import asyncio
import os
import subprocess
import sys
import time

import pytest

from finetune_studio.jobs import launch, store
from finetune_studio.jobs.store import JobSpec


@pytest.fixture(autouse=True)
def _no_supervisor(monkeypatch: pytest.MonkeyPatch, tmp_path) -> None:
    """Never talk to a live supervisor: its socket lives under the (temp) FTS_ROOT. Real job processes here."""
    monkeypatch.setenv("FTS_ROOT", str(tmp_path / "root"))
    monkeypatch.delenv("FTS_JOBS_INPROCESS", raising=False)


def _spec(**params) -> JobSpec:
    gpu = params.pop("gpu", False)
    return JobSpec(id=store.new_id("selftest"), kind="selftest", params=params, gpu=gpu, stop_grace_s=5, title="t")


def test_a_job_runs_to_done_and_leaves_its_state_and_exit_record() -> None:
    spec = _spec(steps=3, delay_s=0.05)
    launch.start(spec)
    d = launch.wait(spec.id, timeout=60)
    assert d["exit"]["status"] == "done" and d["exit"]["code"] == 0
    assert d["state"]["status"] == "done" and d["state"]["step"] == 3
    assert any(j["id"] == spec.id for j in launch.list_jobs("selftest"))


def test_a_failing_job_records_the_error() -> None:
    spec = _spec(steps=3, delay_s=0.05, fail_at=2)
    launch.start(spec)
    ex = launch.wait(spec.id, timeout=60)["exit"]
    assert ex["status"] == "failed" and ex["code"] == 1 and "selftest failure at step 2" in ex["error"]


def test_stop_ends_the_job_cooperatively() -> None:
    spec = _spec(steps=200, delay_s=0.1)
    launch.start(spec)
    deadline = time.monotonic() + 60
    while (launch.describe(spec.id)["state"] or {}).get("step", 0) < 2:
        assert time.monotonic() < deadline
        time.sleep(0.1)
    launch.stop(spec.id)
    d = launch.wait(spec.id, timeout=30)
    assert d["exit"]["status"] == "stopped" and d["exit"]["code"] == 3


def test_a_hard_death_is_reported_as_a_crash_not_running_forever() -> None:
    spec = _spec(steps=5, delay_s=0.05, crash_at=2)
    launch.start(spec)
    d = launch.wait(spec.id, timeout=60)
    assert d["running"] is False and d["exit"]["status"] == "crashed"


def test_the_job_outlives_the_process_that_started_it() -> None:
    """The point of the whole layer: a web restart (its process exiting) must not end the job."""
    spec = _spec(steps=15, delay_s=0.1)
    store.write_spec(spec)
    # A stand-in for the web process: starts the runner the way jobs.launch does, then exits at once.
    code = ("import subprocess, sys; "
            f"subprocess.Popen([sys.executable, '-m', 'finetune_studio.jobs.runner', '--id', '{spec.id}'], "
            "start_new_session=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL); sys.exit(0)")
    parent = subprocess.run([sys.executable, "-c", code], env=dict(os.environ), timeout=60)
    assert parent.returncode == 0                          # the "web process" is gone
    d = launch.wait(spec.id, timeout=60)
    assert d["exit"]["status"] == "done" and d["state"]["step"] == 15


def test_one_gpu_job_at_a_time() -> None:
    first = _spec(steps=100, delay_s=0.1, gpu=True)
    launch.start(first)
    try:
        with pytest.raises(launch.JobRejected) as e:
            launch.start(_spec(steps=1, gpu=True))
        assert e.value.status == 409 and first.id in str(e.value)
        launch.start(_spec(steps=1, delay_s=0.01))           # a CPU job may still run
    finally:
        launch.stop(first.id)
        launch.wait(first.id, timeout=30)


def test_unknown_kind_is_a_bad_spec_exit() -> None:
    spec = JobSpec(id=store.new_id("nope"), kind="nope")
    launch.start(spec)
    ex = launch.wait(spec.id, timeout=60)["exit"]
    assert ex["status"] == "failed" and ex["code"] == 2 and "unknown job kind" in ex["error"]


# ── the supervisor's job table ──────────────────────────────────────────

def test_supervisor_job_table_runs_stops_and_reports_kills(tmp_path) -> None:
    from finetune_studio.supervisor.events import EventLog
    from finetune_studio.supervisor.jobs import JobBusy, JobTable

    async def scenario() -> None:
        events = EventLog(tmp_path / "events.jsonl")
        table = JobTable(events, cwd=os.getcwd())
        ok = _spec(steps=2, delay_s=0.05)
        store.write_spec(ok)
        await table.start(ok.id)
        await asyncio.wait_for(table._jobs[ok.id].waiter, 60)
        assert table.describe(ok.id)["exit"]["status"] == "done"

        gpu = _spec(steps=200, delay_s=0.1, gpu=True)
        store.write_spec(gpu)
        await table.start(gpu.id)
        other = _spec(steps=1, gpu=True)
        store.write_spec(other)
        with pytest.raises(JobBusy):
            await table.start(other.id)
        await table.stop(gpu.id)
        await asyncio.wait_for(table._jobs[gpu.id].waiter, 30)
        assert table.describe(gpu.id)["exit"]["status"] == "stopped"

        crash = _spec(steps=5, delay_s=0.05, crash_at=2)
        store.write_spec(crash)
        await table.start(crash.id)
        await asyncio.wait_for(table._jobs[crash.id].waiter, 60)
        ex = table.describe(crash.id)["exit"]
        assert ex["status"] == "crashed" and ex["signal"] == "SIGKILL"
        kinds = [e["kind"] for e in events.since(0, 100)]
        assert kinds.count("job_started") == 3 and kinds.count("job_exited") == 3

    asyncio.run(scenario())


# ── through a real supervisor: the web component restarts, the job does not notice ──

@pytest.fixture
def short_root(monkeypatch: pytest.MonkeyPatch):
    import shutil
    import tempfile
    d = tempfile.mkdtemp(prefix="fts")                  # AF_UNIX paths are limited to ~100 chars
    monkeypatch.setenv("FTS_ROOT", d)
    yield d
    shutil.rmtree(d, ignore_errors=True)


def test_a_web_restart_under_the_supervisor_leaves_the_job_running(short_root) -> None:
    from finetune_studio.supervisor import control
    from finetune_studio.supervisor.core import Supervisor
    from finetune_studio.supervisor.events import EventLog
    from finetune_studio.supervisor.jobs import JobTable
    from finetune_studio.supervisor.paths import socket_path
    from finetune_studio.supervisor.spec import ComponentSpec

    web = ComponentSpec(name="web", argv=lambda: [sys.executable, "-c", "import time; time.sleep(60)"],
                        ready_after=0.2, grace=2.0, backoff_base=0.2)

    async def main() -> None:
        events = EventLog()
        sup = Supervisor([web], events, jobs=JobTable(events, cwd=os.getcwd()))
        server = await control.serve(sup, socket_path())
        runner = asyncio.create_task(sup.run())
        try:
            spec = _spec(steps=30, delay_s=0.1)
            d = await asyncio.to_thread(launch.start, spec)       # finds the supervisor socket
            assert d["running"] and d["pid"]
            web_pid = sup.snapshot()["components"]["web"]["pid"]
            sup.restart("web")                                      # what `fts restart web` / an update does
            for _ in range(100):
                snap = sup.snapshot()["components"]["web"]
                if snap["pid"] and snap["pid"] != web_pid and snap["state"] == "ready":
                    break
                await asyncio.sleep(0.1)
            assert sup.snapshot()["components"]["web"]["pid"] != web_pid
            assert (await asyncio.to_thread(launch.describe, spec.id))["running"]
            done = await asyncio.to_thread(launch.wait, spec.id, 60)
            assert done["exit"]["status"] == "done" and done["state"]["step"] == 30

            long = _spec(steps=500, delay_s=0.1)
            await asyncio.to_thread(launch.start, long)
            await asyncio.sleep(1.0)
            await sup.shutdown_gracefully()                         # systemctl stop: jobs get their grace
            ex = store.read_exit(long.id)
            assert ex and ex["status"] == "stopped"
        finally:
            sup.shutdown()
            await runner
            await server.stop()

    asyncio.run(main())


def test_the_in_process_executor_runs_the_same_kind_code_and_stops(monkeypatch: pytest.MonkeyPatch) -> None:
    """What the rest of the suite uses (FTS_JOBS_INPROCESS=1): same runner.execute, a thread instead of a pid."""
    monkeypatch.setenv("FTS_JOBS_INPROCESS", "1")
    done = _spec(steps=2, delay_s=0.01)
    launch.start(done)
    d = launch.wait(done.id, timeout=30)
    assert d["exit"]["status"] == "done" and d["state"]["step"] == 2 and store.read_pid(done.id) is None
    slow = _spec(steps=500, delay_s=0.05)
    launch.start(slow)
    assert launch.describe(slow.id)["running"]
    launch.stop(slow.id)
    assert launch.wait(slow.id, timeout=30)["exit"]["status"] == "stopped"


def test_the_web_side_watcher_runs_a_kind_follow_up_once_per_finished_job(monkeypatch: pytest.MonkeyPatch) -> None:
    from finetune_studio.jobs import watch
    monkeypatch.setenv("FTS_JOBS_INPROCESS", "1")
    monkeypatch.setattr(watch, "_HANDLERS", {})
    monkeypatch.setattr(watch, "_seen_running", set())
    monkeypatch.setattr(watch, "_handled", set())
    old = _spec(steps=1, delay_s=0.0)
    launch.start(old)
    launch.wait(old.id, timeout=30)
    seen: list[str] = []
    watch.on_finished("selftest", lambda j: seen.append(j["id"]))
    assert watch.poll_once(first=True) == [] and seen == []          # ended before this web process: recorded only
    job = _spec(steps=40, delay_s=0.02)
    launch.start(job)
    watch.poll_once()                                                 # seen running
    launch.wait(job.id, timeout=30)
    assert watch.poll_once() == [job.id] and seen == [job.id]
    assert watch.poll_once() == [] and seen == [job.id]


def test_a_finished_gpu_job_that_released_the_card_does_not_block_the_next(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("FTS_JOBS_INPROCESS", "1")
    first = _spec(steps=300, delay_s=0.05, gpu=True)
    launch.start(first)
    with pytest.raises(launch.JobRejected):
        launch.start(_spec(steps=1, gpu=True))
    store.release_gpu(first.id)                                      # past its GPU work, only its exit is left
    second = _spec(steps=1, delay_s=0.0, gpu=True)
    launch.start(second)
    launch.wait(second.id, timeout=30)
    launch.stop(first.id)
    launch.wait(first.id, timeout=30)
