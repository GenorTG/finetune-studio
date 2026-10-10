"""Control API: JSON over HTTP on a unix socket (mode 0600 dir; no TCP port, no token)."""
from __future__ import annotations

import asyncio
import contextlib
import os
from pathlib import Path

import uvicorn
from starlette.applications import Starlette
from starlette.requests import Request
from starlette.responses import JSONResponse
from starlette.routing import Route

from finetune_studio.supervisor.core import Supervisor, UnknownComponent

ACTIONS = ("start", "stop", "restart")


def build_app(sup: Supervisor) -> Starlette:
    async def status(_: Request) -> JSONResponse:
        return JSONResponse(sup.snapshot())

    async def events(request: Request) -> JSONResponse:
        try:
            since = int(request.query_params.get("since", "0"))
            limit = max(1, min(int(request.query_params.get("limit", "200")), 500))
        except ValueError:
            return JSONResponse({"error": "since and limit must be integers"}, status_code=400)
        return JSONResponse({"events": sup.events.since(since, limit)})

    async def logs(request: Request) -> JSONResponse:
        try:
            lines = max(1, min(int(request.query_params.get("lines", "100")), 400))
            return JSONResponse({"lines": sup.logs(request.path_params["name"], lines)})
        except ValueError:
            return JSONResponse({"error": "lines must be an integer"}, status_code=400)
        except UnknownComponent:
            return JSONResponse({"error": f"unknown component {request.path_params['name']!r}"}, status_code=404)

    async def act(request: Request) -> JSONResponse:
        name, action = request.path_params["name"], request.path_params["action"]
        if action not in ACTIONS:
            return JSONResponse({"error": f"unknown action {action!r}; use {', '.join(ACTIONS)}"}, status_code=404)
        try:
            getattr(sup, action)(name)
        except UnknownComponent:
            return JSONResponse({"error": f"unknown component {name!r}"}, status_code=404)
        return JSONResponse({"component": name, "action": action, "accepted": True}, status_code=202)

    async def shutdown(_: Request) -> JSONResponse:
        sup.events.emit("supervisor", "shutdown_requested")
        sup.shutdown()
        return JSONResponse({"accepted": True}, status_code=202)

    return Starlette(routes=[
        Route("/v1/status", status),
        Route("/v1/shutdown", shutdown, methods=["POST"]),
        Route("/v1/events", events),
        Route("/v1/components/{name}/logs", logs),
        Route("/v1/components/{name}/{action}", act, methods=["POST"]),
    ])


class ControlServer:
    def __init__(self, server: uvicorn.Server, task: asyncio.Task[None]) -> None:
        self._server, self._task = server, task

    async def stop(self) -> None:
        self._server.should_exit = True
        await self._task


async def serve(sup: Supervisor, path: Path) -> ControlServer:
    """Start the control server on ``path``; returns once it accepts connections."""
    path.parent.mkdir(parents=True, exist_ok=True)
    os.chmod(path.parent, 0o700)
    with contextlib.suppress(FileNotFoundError):
        path.unlink()
    server = uvicorn.Server(uvicorn.Config(build_app(sup), uds=str(path), log_level="warning", lifespan="off"))
    server.install_signal_handlers = lambda: None  # type: ignore[method-assign]  # the supervisor owns signals
    task = asyncio.create_task(server.serve(), name="supervisor-control")
    while not server.started:
        if task.done():
            task.result()
            raise RuntimeError("control server stopped during startup")
        await asyncio.sleep(0.02)
    with contextlib.suppress(OSError):
        os.chmod(path, 0o600)
    return ControlServer(server, task)
