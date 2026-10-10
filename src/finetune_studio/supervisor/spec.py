"""Component description and state names for the supervisor."""
from __future__ import annotations

from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from enum import StrEnum

# A probe returns (healthy, short detail). It must bound its own time.
Probe = Callable[[], Awaitable[tuple[bool, str]]]


class State(StrEnum):
    STOPPED = "stopped"
    STARTING = "starting"
    READY = "ready"
    UNHEALTHY = "unhealthy"
    BACKOFF = "backoff"
    FAILED = "failed"


@dataclass
class ComponentSpec:
    """One managed child process and its restart/health policy."""

    name: str
    argv: Callable[[], list[str]]
    probe: Probe | None = None
    env: dict[str, str] = field(default_factory=dict)
    cwd: str | None = None
    pass_fds: Callable[[], tuple[int, ...]] = lambda: ()
    prepare: Callable[[], None] | None = None  # runs before each spawn (e.g. open the listen socket)
    release: Callable[[], None] | None = None  # runs when the component is stopped/failed for good
    start_timeout: float = 180.0  # STARTING -> READY deadline (the web app scans models at start)
    ready_after: float = 1.0  # no probe: alive this long means ready
    probe_interval: float = 5.0
    probe_failures: int = 3  # consecutive failed probes before a restart
    grace: float = 15.0  # SIGTERM -> SIGKILL
    backoff_base: float = 1.0
    backoff_max: float = 30.0
    stable_after: float = 60.0  # ran this long: the crash counter resets
    max_fast_crashes: int = 8  # consecutive early deaths before FAILED (no more restarts until told)
