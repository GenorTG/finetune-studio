"""Supervisor core against REAL child processes: ready, crash+backoff, FAILED, health restart, stop, shutdown."""
from __future__ import annotations

import asyncio
import os
import signal
import sys
import time

from finetune_studio.supervisor.core import Supervisor
from finetune_studio.supervisor.events import EventLog
from finetune_studio.supervisor.spec import ComponentSpec


def spec(code: str, **kw) -> ComponentSpec:
    base = {"name": "c", "argv": lambda: [sys.executable, "-u", "-c", code], "ready_after": 0.3, "grace": 2.0,
            "backoff_base": 0.2, "backoff_max": 0.5, "stable_after": 30.0, "max_fast_crashes": 4}
    base.update(kw)
    return ComponentSpec(**base)


async def until(pred, timeout: float = 10.0):
    end = time.monotonic() + timeout
    while time.monotonic() < end:
        if pred():
            return
        await asyncio.sleep(0.05)
    raise AssertionError("condition not reached in time")


def drive(spec_: ComponentSpec, scenario):
    """Run `scenario(sup)` while the supervisor supervises, then shut down."""
    async def main():
        sup = Supervisor([spec_], EventLog())
        runner = asyncio.create_task(sup.run())
        try:
            await scenario(sup)
        finally:
            sup.shutdown()
            await asyncio.wait_for(runner, 15)
        return sup
    return asyncio.run(main())


def state(sup: Supervisor) -> str:
    return sup.snapshot()["components"]["c"]["state"]


def kinds(sup: Supervisor) -> list[str]:
    return [e["kind"] for e in sup.events.since(0)]


def test_alive_child_becomes_ready_and_stops_on_request():
    async def scenario(sup):
        await until(lambda: state(sup) == "ready")
        snap = sup.snapshot()["components"]["c"]
        assert snap["pid"] and snap["restarts"] == 0
        sup.stop("c")
        await until(lambda: state(sup) == "stopped")
        assert sup.snapshot()["components"]["c"]["pid"] is None

    sup = drive(spec("import time; time.sleep(60)"), scenario)
    assert "spawned" in kinds(sup) and "ready" in kinds(sup) and "stop_requested" in kinds(sup)


def test_native_abort_is_reported_by_signal_and_restarted_with_output_tail():
    code = "import os, signal; print('about to die', flush=True); os.kill(os.getpid(), signal.SIGABRT)"

    async def scenario(sup):
        await until(lambda: sup.snapshot()["components"]["c"]["spawns"] >= 2)
        last = sup.snapshot()["components"]["c"]["last_exit"]
        assert last["signal"] == "SIGABRT" and last["code"] is None
        assert "about to die" in last["tail"]

    sup = drive(spec(code), scenario)
    exited = [e for e in sup.events.since(0) if e["kind"] == "exited"]
    assert exited and exited[0]["signal"] == "SIGABRT"
    assert "backoff" in kinds(sup)


def test_repeated_early_deaths_end_in_failed_and_manual_restart_recovers():
    async def scenario(sup):
        await until(lambda: state(sup) == "failed", 15)
        spawns = sup.snapshot()["components"]["c"]["spawns"]
        await asyncio.sleep(0.8)
        assert sup.snapshot()["components"]["c"]["spawns"] == spawns  # no more restarts on its own
        sup.restart("c")
        await until(lambda: sup.snapshot()["components"]["c"]["spawns"] > spawns)

    sup = drive(spec("import sys; sys.exit(3)"), scenario)
    assert "failed" in kinds(sup)


def test_failed_health_probe_restarts_a_hung_child():
    calls = {"n": 0}

    async def probe():
        calls["n"] += 1
        # Healthy for the first few probes of the first process, then permanently failing.
        return (calls["n"] <= 2), "stub"

    async def scenario(sup):
        await until(lambda: "unhealthy" in kinds(sup), 15)
        await until(lambda: sup.snapshot()["components"]["c"]["spawns"] >= 2, 15)

    drive(spec("import time; time.sleep(60)", probe=probe, probe_interval=0.1, probe_failures=2), scenario)


def test_child_that_never_gets_healthy_hits_start_timeout():
    async def never():
        return False, "refused"

    async def scenario(sup):
        await until(lambda: "start_timeout" in kinds(sup), 15)

    drive(spec("import time; time.sleep(60)", probe=never, start_timeout=0.5), scenario)


def test_requested_restart_has_no_backoff_penalty_and_new_pid():
    async def scenario(sup):
        await until(lambda: state(sup) == "ready")
        first = sup.snapshot()["components"]["c"]["pid"]
        sup.restart("c")
        await until(lambda: sup.snapshot()["components"]["c"]["spawns"] == 2 and state(sup) == "ready")
        snap = sup.snapshot()["components"]["c"]
        assert snap["pid"] != first and snap["consecutive_crashes"] == 0

    drive(spec("import time; time.sleep(60)"), scenario)


def test_shutdown_kills_the_whole_process_group_including_grandchildren(tmp_path):
    pidfile = tmp_path / "grandchild.pid"
    code = (
        "import subprocess, sys, time\n"
        f"p = subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(120)'])\n"
        f"open({str(pidfile)!r}, 'w').write(str(p.pid))\n"
        "time.sleep(120)\n"
    )
    seen = {}

    async def scenario(sup):
        await until(lambda: pidfile.exists() and pidfile.read_text() != "")
        seen["pid"] = int(pidfile.read_text())
        os.kill(seen["pid"], 0)  # alive

    drive(spec(code), scenario)
    deadline = time.monotonic() + 5
    while time.monotonic() < deadline:
        try:
            os.kill(seen["pid"], 0)
        except ProcessLookupError:
            return
        time.sleep(0.05)
    os.kill(seen["pid"], signal.SIGKILL)
    raise AssertionError("grandchild survived supervisor shutdown")


def test_sigterm_ignoring_child_is_killed_after_grace():
    code = "import signal, time; signal.signal(signal.SIGTERM, signal.SIG_IGN); time.sleep(60)"

    async def scenario(sup):
        await until(lambda: state(sup) == "ready")
        sup.stop("c")
        await until(lambda: state(sup) == "stopped", 8)

    sup = drive(spec(code, grace=0.5), scenario)
    assert "kill_escalation" in kinds(sup)


def test_unknown_component_raises_keyerror():
    sup = Supervisor([spec("pass")])
    try:
        sup.restart("nope")
    except KeyError:
        return
    raise AssertionError("expected KeyError")


def test_event_log_ring_and_rotation(tmp_path):
    log = EventLog(tmp_path / "events.jsonl", max_bytes=300)
    for i in range(20):
        log.emit("c", "tick", i=i)
    assert [e["seq"] for e in log.since(18)] == [19, 20]
    assert (tmp_path / "events.jsonl.1").exists()
