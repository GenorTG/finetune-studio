"""``/api/service/*`` — what runs the app, whether it is healthy, and what went wrong.

Reads the supervisor over its control socket. Honest statuses: ``managed: false`` (200) when the app
was started without a supervisor; 503 when a supervisor socket exists but does not answer; 409 for an
action that needs a supervisor; 404 for an unknown component.
"""
from __future__ import annotations

import asyncio
import time
from typing import Any

from fastapi import APIRouter, Query
from fastapi.responses import JSONResponse

from finetune_studio.supervisor.client import (
    SupervisorClient,
    SupervisorError,
    SupervisorUnavailable,
)
from finetune_studio.supervisor.paths import socket_path

router = APIRouter(prefix="/api/service", tags=["service"])

NOTICE_WINDOW_S = 600.0  # how long an unexpected exit is shown as "the app restarted"
ACTIONS = ("start", "stop", "restart")


def _exit_text(last: dict[str, Any]) -> str:
    if last.get("reason") in ("unhealthy", "start_timeout"):
        return "stopped answering health checks and was killed"
    if last.get("signal"):
        return f"killed by {last['signal']}"
    if last.get("code") is not None:
        return f"exited with code {last['code']}"
    return str(last.get("error") or "exited")


def find_problems(components: dict[str, Any], now: float | None = None) -> list[dict[str, Any]]:
    """Plain-language problems for the banner/card, worst first. Pure function of the component table."""
    now = now or time.time()
    out: list[dict[str, Any]] = []
    for name, c in components.items():
        last = c.get("last_exit") or {}
        why = f" Last exit: {_exit_text(last)}." if last else ""
        state = c["state"]
        if state == "failed":
            out.append({"id": f"{name}:failed", "severity": "error", "component": name,
                        "title": f"{name} keeps crashing and was stopped",
                        "detail": f"{c['consecutive_crashes']} early exits in a row, automatic restarts are off.{why} Check the log, then restart it."})
        elif state in ("backoff", "unhealthy"):
            out.append({"id": f"{name}:{state}", "severity": "warning", "component": name,
                        "title": f"{name} is restarting" if state == "backoff" else f"{name} is not answering",
                        "detail": f"Retrying in {c.get('retry_in_s')} s.{why}" if state == "backoff" else "Health checks are failing."})
        elif state == "starting" and c.get("restarts"):
            out.append({"id": f"{name}:starting", "severity": "warning", "component": name,
                        "title": f"{name} is starting again", "detail": f"Back in a moment.{why}"})
        elif state == "ready" and last and last.get("reason") != "requested" and now - float(last.get("at") or 0) < NOTICE_WINDOW_S:
            out.append({"id": f"{name}:exit:{last.get('at')}", "severity": "notice", "component": name,
                        "title": f"{name} restarted unexpectedly",
                        "detail": f"The previous process was {_exit_text(last)}. Jobs that were running were stopped and are marked failed; a loaded model was unloaded."})
    order = {"error": 0, "warning": 1, "notice": 2}
    return sorted(out, key=lambda p: order[p["severity"]])


async def _call(fn, *args):
    return await asyncio.to_thread(fn, *args)


def _unmanaged() -> dict[str, Any]:
    return {"managed": False, "reachable": False, "components": {}, "problems": [],
            "hint": "This server was started without a supervisor (run.sh, fts webui or plain uvicorn). Start it with `fts up` or `bash install-service.sh` to get restarts and health checks."}


@router.get("/status")
async def status() -> JSONResponse:
    if not socket_path().exists():
        return JSONResponse(_unmanaged())
    try:
        snap = await _call(SupervisorClient(timeout=2.0).status)
    except SupervisorUnavailable as exc:
        return JSONResponse({"managed": True, "reachable": False, "components": {}, "error": str(exc),
                             "problems": [{"id": "supervisor:unreachable", "severity": "error", "component": "supervisor",
                                           "title": "The supervisor does not answer", "detail": str(exc)}]}, status_code=503)
    return JSONResponse({"managed": True, "reachable": True, **snap, "problems": find_problems(snap["components"])})


@router.get("/events")
async def events(since: int = Query(0, ge=0), limit: int = Query(50, ge=1, le=500)) -> JSONResponse:
    if not socket_path().exists():
        return JSONResponse({"error": "no supervisor"}, status_code=409)
    try:
        return JSONResponse({"events": await _call(SupervisorClient(timeout=2.0).events, since, limit)})
    except SupervisorUnavailable as exc:
        return JSONResponse({"error": str(exc)}, status_code=503)


@router.get("/logs/{name}")
async def logs(name: str, lines: int = Query(100, ge=1, le=400)) -> JSONResponse:
    if not socket_path().exists():
        return JSONResponse({"error": "no supervisor"}, status_code=409)
    try:
        return JSONResponse({"lines": await _call(SupervisorClient(timeout=2.0).logs, name, lines)})
    except SupervisorUnavailable as exc:
        return JSONResponse({"error": str(exc)}, status_code=503)
    except SupervisorError as exc:
        return JSONResponse({"error": str(exc)}, status_code=exc.status)


@router.post("/components/{name}/{action}")
async def act(name: str, action: str) -> JSONResponse:
    if action not in ACTIONS:
        return JSONResponse({"error": f"unknown action {action!r}; use {', '.join(ACTIONS)}"}, status_code=404)
    if not socket_path().exists():
        return JSONResponse({"error": "This server runs without a supervisor, so it cannot restart itself. Start it with `fts up`."}, status_code=409)
    try:
        return JSONResponse(await _call(SupervisorClient(timeout=3.0).act, name, action), status_code=202)
    except SupervisorUnavailable as exc:
        return JSONResponse({"error": str(exc)}, status_code=503)
    except SupervisorError as exc:
        return JSONResponse({"error": str(exc)}, status_code=exc.status)
