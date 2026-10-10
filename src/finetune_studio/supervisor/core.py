"""The supervisor: desired-state table, process table, health probes, restart policy.

One asyncio task per component owns that component's lifecycle. The only way to change a
component is ``start`` / ``stop`` / ``restart`` (the control API calls these); everything the
supervisor observes is recorded as an event.
"""
from __future__ import annotations

import asyncio
import contextlib
import os
import signal
import sys
import time
from collections import deque
from dataclasses import dataclass, field
from typing import Any

from finetune_studio.supervisor.env import child_env
from finetune_studio.supervisor.events import EventLog
from finetune_studio.supervisor.spec import ComponentSpec, State

TICK = 0.2
LOG_RING = 400
TAIL_LINES = 25


def _log(msg: str) -> None:
    print(f"[supervisor] {msg}", flush=True)


def describe_exit(returncode: int) -> dict[str, Any]:
    """Exit code or terminating signal, by name."""
    if returncode < 0:
        try:
            name = signal.Signals(-returncode).name
        except ValueError:
            name = f"SIG{-returncode}"
        return {"code": None, "signal": name}
    return {"code": returncode, "signal": None}


@dataclass
class _Component:
    spec: ComponentSpec
    desired: str = "running"
    state: State = State.STOPPED
    proc: asyncio.subprocess.Process | None = None
    started_at: float = 0.0  # monotonic
    started_wall: float = 0.0
    spawns: int = 0
    crashes: int = 0  # consecutive early deaths
    last_exit: dict[str, Any] | None = None
    health: dict[str, Any] = field(default_factory=dict)
    backoff_until: float = 0.0
    restart_requested: bool = False
    logs: deque[str] = field(default_factory=lambda: deque(maxlen=LOG_RING))
    wake: asyncio.Event = field(default_factory=asyncio.Event)
    task: asyncio.Task[None] | None = None


class UnknownComponent(KeyError):
    pass


class Supervisor:
    def __init__(self, specs: list[ComponentSpec], events: EventLog | None = None) -> None:
        self.events = events or EventLog()
        self.started_at = time.time()
        self._components = {s.name: _Component(spec=s) for s in specs}
        self._stopping = False

    # ── lifecycle ──────────────────────────────────────────────────────

    async def run(self) -> None:
        """Supervise every component until :meth:`shutdown` is called."""
        for c in self._components.values():
            c.task = asyncio.create_task(self._loop(c), name=f"supervise-{c.spec.name}")
        await asyncio.gather(*(c.task for c in self._components.values() if c.task))

    def shutdown(self) -> None:
        self._stopping = True
        for c in self._components.values():
            c.wake.set()

    # ── operations (control API) ───────────────────────────────────────

    def start(self, name: str) -> None:
        c = self._get(name)
        c.desired = "running"
        c.crashes = 0
        self.events.emit(name, "start_requested")
        c.wake.set()

    def stop(self, name: str) -> None:
        c = self._get(name)
        c.desired = "stopped"
        self.events.emit(name, "stop_requested")
        c.wake.set()

    def restart(self, name: str) -> None:
        c = self._get(name)
        c.desired = "running"
        c.crashes = 0
        c.restart_requested = True
        self.events.emit(name, "restart_requested")
        c.wake.set()

    def logs(self, name: str, lines: int = 100) -> list[str]:
        return list(self._get(name).logs)[-lines:]

    def snapshot(self) -> dict[str, Any]:
        now = time.time()
        comps = {}
        for name, c in self._components.items():
            running = c.proc is not None and c.proc.returncode is None
            comps[name] = {
                "desired": c.desired,
                "state": c.state.value,
                "pid": c.proc.pid if running and c.proc else None,
                "uptime_s": round(now - c.started_wall, 1) if running else None,
                "spawns": c.spawns,
                "restarts": max(0, c.spawns - 1),
                "consecutive_crashes": c.crashes,
                "last_exit": c.last_exit,
                "health": dict(c.health),
                "retry_in_s": round(max(0.0, c.backoff_until - time.monotonic()), 1) if c.state is State.BACKOFF else None,
            }
        return {"supervisor": {"pid": os.getpid(), "started_at": self.started_at, "uptime_s": round(now - self.started_at, 1)},
                "components": comps}

    def _get(self, name: str) -> _Component:
        try:
            return self._components[name]
        except KeyError:
            raise UnknownComponent(name) from None

    # ── per-component loop ─────────────────────────────────────────────

    async def _loop(self, c: _Component) -> None:
        name = c.spec.name
        while not self._stopping:
            if c.desired != "running":
                c.state = State.FAILED if c.desired == "failed" else State.STOPPED
                self._release(c)
                await self._sleep(c, None)
                continue
            reason = await self._run_once(c)
            if self._stopping or c.desired != "running" or reason == "requested":
                continue
            # The child died, hung or failed its probe on its own: restart with backoff.
            ran = time.monotonic() - c.started_at
            c.crashes = 0 if ran >= c.spec.stable_after else c.crashes + 1
            if c.crashes >= c.spec.max_fast_crashes:
                c.state = State.FAILED
                self._release(c)
                self.events.emit(name, "failed", crashes=c.crashes, detail="too many early deaths; restarts stopped")
                _log(f"{name}: FAILED after {c.crashes} early deaths; use 'fts restart {name}'")
                c.desired = "failed"
                await self._sleep(c, None)
                continue
            delay = min(c.spec.backoff_max, c.spec.backoff_base * (2 ** max(0, c.crashes - 1)))
            c.state = State.BACKOFF
            c.backoff_until = time.monotonic() + delay
            self.events.emit(name, "backoff", delay_s=delay, reason=reason, crashes=c.crashes)
            await self._sleep(c, delay)
        await self._terminate(c)
        c.state = State.STOPPED
        self._release(c)

    async def _sleep(self, c: _Component, timeout: float | None) -> None:
        """Wait for a wake-up (operator action or shutdown) or the timeout."""
        with contextlib.suppress(TimeoutError):
            await asyncio.wait_for(c.wake.wait(), timeout)
        c.wake.clear()

    async def _run_once(self, c: _Component) -> str:
        spec = c.spec
        name = spec.name
        try:
            if spec.prepare:
                spec.prepare()
            env = child_env(spec.env)
            c.proc = await asyncio.create_subprocess_exec(
                *spec.argv(), cwd=spec.cwd, env=env, stdin=asyncio.subprocess.DEVNULL,
                stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.STDOUT,
                start_new_session=True, pass_fds=spec.pass_fds(),
            )
        except Exception as exc:  # noqa: BLE001 - spawn failure is a reportable event, not a crash
            c.last_exit = {"code": None, "signal": None, "error": f"spawn failed: {exc}", "at": time.time()}
            self.events.emit(name, "spawn_failed", error=str(exc))
            _log(f"{name}: spawn failed: {exc}")
            c.started_at = time.monotonic()
            return "spawn_failed"
        proc = c.proc
        c.spawns += 1
        c.started_at = time.monotonic()
        c.started_wall = time.time()
        c.state = State.STARTING
        c.health = {}
        c.restart_requested = False
        self.events.emit(name, "spawned", pid=proc.pid, spawns=c.spawns)
        _log(f"{name}: started pid {proc.pid}")
        reader = asyncio.create_task(self._pump(c, proc))
        next_probe = time.monotonic()
        fails = 0
        reason = "exited"
        try:
            while True:
                await self._sleep(c, TICK)
                if proc.returncode is not None:
                    break
                if self._stopping or c.desired != "running":
                    reason = "requested"
                    break
                if c.restart_requested:
                    reason = "requested"
                    break
                now = time.monotonic()
                alive_for = now - c.started_at
                if spec.probe is None:
                    if c.state is State.STARTING and alive_for >= spec.ready_after:
                        self._mark_ready(c, alive_for)
                    continue
                if now < next_probe:
                    continue
                next_probe = now + (spec.probe_interval if c.state is State.READY else 1.0)
                ok, detail = await spec.probe()
                alive_for = time.monotonic() - c.started_at
                c.health = {"ok": ok, "detail": detail, "checked_at": time.time(), "consecutive_failures": 0 if ok else fails + 1}
                if ok:
                    fails = 0
                    if c.state is not State.READY:
                        self._mark_ready(c, alive_for)
                    continue
                if c.state is State.STARTING:
                    if alive_for > spec.start_timeout:
                        self.events.emit(name, "start_timeout", after_s=round(alive_for), detail=detail)
                        reason = "start_timeout"
                        break
                    continue
                fails += 1
                if fails >= spec.probe_failures:
                    c.state = State.UNHEALTHY
                    self.events.emit(name, "unhealthy", detail=detail, failures=fails)
                    reason = "unhealthy"
                    break
        finally:
            if proc.returncode is None:
                await self._terminate(c)
            with contextlib.suppress(Exception):
                await asyncio.wait_for(reader, 2)
        self._record_exit(c, reason)
        return reason

    def _mark_ready(self, c: _Component, after: float) -> None:
        c.state = State.READY
        self.events.emit(c.spec.name, "ready", after_s=round(after, 1))
        _log(f"{c.spec.name}: ready after {after:.1f}s")

    def _record_exit(self, c: _Component, reason: str) -> None:
        proc = c.proc
        if proc is None or proc.returncode is None:
            return
        info = describe_exit(proc.returncode)
        tail = list(c.logs)[-TAIL_LINES:]
        c.last_exit = {**info, "reason": reason, "at": time.time(), "ran_s": round(time.monotonic() - c.started_at, 1)}
        clean = reason == "requested"
        c.last_exit["tail"] = [] if clean else tail
        self.events.emit(c.spec.name, "exited", pid=proc.pid, **info, reason=reason, tail=c.last_exit["tail"])
        _log(f"{c.spec.name}: exited ({info['signal'] or 'code ' + str(info['code'])}), reason={reason}")

    async def _pump(self, c: _Component, proc: asyncio.subprocess.Process) -> None:
        """Forward the child's output to our stdout (the journal) and keep a tail for diagnostics."""
        assert proc.stdout is not None
        async for raw in proc.stdout:
            line = raw.decode("utf-8", "replace").rstrip("\n")
            c.logs.append(line)
            sys.stdout.write(line + "\n")
            sys.stdout.flush()

    async def _terminate(self, c: _Component) -> None:
        """SIGTERM the child's whole process group, SIGKILL after the grace period."""
        proc = c.proc
        if proc is None or proc.returncode is not None:
            return
        for sig, wait in ((signal.SIGTERM, c.spec.grace), (signal.SIGKILL, 5.0)):
            with contextlib.suppress(ProcessLookupError):
                os.killpg(proc.pid, sig)
            try:
                await asyncio.wait_for(proc.wait(), wait)
                break
            except TimeoutError:
                self.events.emit(c.spec.name, "kill_escalation", signal=sig.name)
        # The leader is gone; sweep stragglers (e.g. a training worker) from the group.
        with contextlib.suppress(ProcessLookupError, PermissionError):
            os.killpg(proc.pid, signal.SIGKILL)

    def _release(self, c: _Component) -> None:
        if c.spec.release:
            with contextlib.suppress(Exception):
                c.spec.release()
