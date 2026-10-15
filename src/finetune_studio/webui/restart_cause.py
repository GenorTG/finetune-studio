"""Why did this web process start? Asks the supervisor so interrupted jobs get an honest reason."""
from __future__ import annotations

import time
from typing import Any

RECENT_S = 300.0  # an exit older than this is not what interrupted the jobs found at start-up


def describe(last_exit: dict[str, Any] | None, now: float | None = None) -> str:
    """A sentence fragment for the previous web process's exit, or "" when unknown or old."""
    if not last_exit or (now or time.time()) - float(last_exit.get("at") or 0) > RECENT_S:
        return ""
    if last_exit.get("reason") == "requested":
        return " (the server was restarted on request)"
    if last_exit.get("reason") in ("unhealthy", "start_timeout"):
        return " (the server stopped answering health checks and was restarted by the supervisor)"
    if last_exit.get("signal"):
        return f" (the server process died with {last_exit['signal']} and was restarted by the supervisor)"
    if last_exit.get("code") is not None:
        return f" (the server process exited with code {last_exit['code']} and was restarted by the supervisor)"
    return ""


def restart_cause() -> str:
    """Cause text for the current start-up; "" when not supervised or nothing recent."""
    from finetune_studio.supervisor.client import (
        SupervisorClient,
        SupervisorError,
        SupervisorUnavailable,
    )

    try:
        comps = SupervisorClient(timeout=1.5).status()["components"]
    except (SupervisorUnavailable, SupervisorError, KeyError):
        return ""
    return describe((comps.get("web") or {}).get("last_exit"))
