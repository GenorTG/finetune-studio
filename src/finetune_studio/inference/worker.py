"""``python -m finetune_studio.inference.worker --socket PATH``: the process that holds the model.

One engine, one model, one request at a time (a worker-side lock serializes load/unload/generate: the
same rule ``ENGINE_LOCK`` enforced in-process). Errors are honest HTTP statuses: 409 when no model is
loaded, 500 with the exception text when a load or a generation fails. A native abort (CUDA, llama.cpp)
kills this process only; the supervisor (or the launcher) starts a new, empty one.
"""
from __future__ import annotations

import argparse
import asyncio
import contextlib
import os
import signal
import sys
import threading
import time
from pathlib import Path
from typing import Any

import uvicorn
from starlette.applications import Starlette
from starlette.requests import Request
from starlette.responses import JSONResponse
from starlette.routing import Route

PARENT_POLL_S = 2.0


def snapshot(engine: Any) -> dict[str, Any]:
    loaded = engine.model is not None
    return {
        "loaded": loaded,
        "model_path": engine.model_path if loaded else None,
        "is_gguf": bool(engine.is_gguf) if loaded else False,
        "vision": bool(getattr(engine, "vision", False)) if loaded else False,
        "mmproj_path": getattr(engine, "mmproj_path", None) if loaded else None,
        "n_ctx": getattr(engine, "n_ctx", None) if loaded else None,
        "n_gpu_layers": getattr(engine, "n_gpu_layers", None) if loaded else None,
        "offload": dict(getattr(engine, "offload", None) or {}) if loaded else {},
        "idle_seconds": int(getattr(engine, "idle_seconds", 0) or 0),
        "busy": int(getattr(engine, "_busy", 0) or 0),
        "last_used": float(getattr(engine, "_last_used", 0.0) or 0.0),
        "loading_path": getattr(engine, "_loading_path", None),
        "loading_started": float(getattr(engine, "_loading_started", 0.0) or 0.0),
        "pid": os.getpid(),
    }


def build_app(engine: Any) -> Starlette:
    lock = threading.Lock()

    def _error(exc: BaseException, status: int = 500) -> JSONResponse:
        return JSONResponse({"detail": f"{type(exc).__name__}: {exc}"}, status_code=status)

    async def health(_: Request) -> JSONResponse:
        return JSONResponse({"status": "ok", "pid": os.getpid()})

    async def status(_: Request) -> JSONResponse:
        return JSONResponse(snapshot(engine))

    async def load(request: Request) -> JSONResponse:
        body = await request.json()
        path = body.pop("model_path", None)
        if not path:
            return JSONResponse({"detail": "model_path is required"}, status_code=400)

        def _do() -> None:
            with lock:
                engine.load(path, **body)

        try:
            await asyncio.to_thread(_do)
        except Exception as exc:  # noqa: BLE001 - the caller gets the real reason
            return _error(exc)
        return JSONResponse(snapshot(engine))

    async def unload(_: Request) -> JSONResponse:
        def _do() -> None:
            with lock:
                engine.unload()

        try:
            await asyncio.to_thread(_do)
        except Exception as exc:  # noqa: BLE001
            return _error(exc)
        return JSONResponse(snapshot(engine))

    async def generate(request: Request) -> JSONResponse:
        body = await request.json()
        messages = body.pop("messages", None)
        if messages is None:
            return JSONResponse({"detail": "messages is required"}, status_code=400)
        if engine.model is None:
            return JSONResponse({"detail": "No model loaded"}, status_code=409)

        def _do() -> str:
            with lock:
                return engine.generate(messages, **body)

        try:
            text = await asyncio.to_thread(_do)
        except Exception as exc:  # noqa: BLE001
            return _error(exc)
        return JSONResponse({"text": text, **snapshot(engine)})

    async def chat_completion(request: Request) -> JSONResponse:
        body = await request.json()
        model = engine.model
        if model is None:
            return JSONResponse({"detail": "No model loaded"}, status_code=409)
        if not hasattr(model, "create_chat_completion"):
            return JSONResponse({"detail": "the loaded model is not a llama.cpp model (no create_chat_completion)"}, status_code=409)

        def _do() -> Any:
            with lock:
                return model.create_chat_completion(**body)

        try:
            result = await asyncio.to_thread(_do)
        except Exception as exc:  # noqa: BLE001
            return _error(exc)
        return JSONResponse(result)

    return Starlette(routes=[
        Route("/v1/health", health),
        Route("/v1/status", status),
        Route("/v1/load", load, methods=["POST"]),
        Route("/v1/unload", unload, methods=["POST"]),
        Route("/v1/generate", generate, methods=["POST"]),
        Route("/v1/chat_completion", chat_completion, methods=["POST"]),
    ])


def make_engine(kind: str) -> Any:
    if kind == "fake":
        from finetune_studio.inference.fake import FakeEngine
        return FakeEngine()
    from finetune_studio.testing.inference import InferenceEngine
    return InferenceEngine()


def watch_parent(owner: int | None = None) -> None:
    """Exit when the process that started us is gone (standalone web server, CLI command, supervisor).

    ``owner``: the pid that owns this worker. On Windows the direct parent can be a venv's python.exe launcher,
    which lives exactly as long as this process, and Windows never reparents (getppid() keeps a dead parent's pid),
    so killing the web app left the worker holding the model (VM 2026-10-22). The launcher passes its own pid.
    """
    parent = os.getppid()
    try:
        import psutil
        exists = psutil.pid_exists
    except ImportError:
        def exists(_pid: int) -> bool:
            return True

    def loop() -> None:
        while True:
            time.sleep(PARENT_POLL_S)
            if os.getppid() != parent or not exists(owner or parent):
                with contextlib.suppress(Exception):     # stdout may be the dead parent's pipe (BrokenPipeError)
                    print("[inference] parent process gone; exiting", flush=True)
                os._exit(0)

    threading.Thread(target=loop, name="parent-watch", daemon=True).start()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m finetune_studio.inference.worker")
    parser.add_argument("--socket", required=True)
    parser.add_argument("--engine", choices=["real", "fake"], default="real", help="fake: test seam, no GPU, no model file")
    parser.add_argument("--owner-pid", type=int, default=None, help="exit when this pid is gone (default: the parent)")
    args = parser.parse_args(argv)
    from finetune_studio import logsetup
    logsetup.setup("inference")
    sock = Path(args.socket)
    sock.parent.mkdir(parents=True, exist_ok=True)
    with contextlib.suppress(OSError):
        os.chmod(sock.parent, 0o700)
    with contextlib.suppress(FileNotFoundError):
        sock.unlink()
    engine = make_engine(args.engine)
    watch_parent(args.owner_pid)
    from finetune_studio.supervisor.client import (
        HAS_AF_UNIX,
        bind_loopback,
        port_file,
        require_token,
    )

    app: Any = build_app(engine)
    listen: list[Any] | None = None
    if HAS_AF_UNIX:
        config = uvicorn.Config(app, uds=str(sock), log_level="warning", lifespan="off", timeout_graceful_shutdown=3)
    else:
        lsock, token = bind_loopback(sock)
        listen = [lsock]
        config = uvicorn.Config(require_token(app, token), log_level="warning", lifespan="off",
                                timeout_graceful_shutdown=3)
    print(f"[inference] pid {os.getpid()} engine={args.engine} socket={sock}"
          + ("" if HAS_AF_UNIX else f" (loopback port {listen[0].getsockname()[1]})"), flush=True)
    server = uvicorn.Server(config)

    async def serve() -> None:
        # Own the signals: uvicorn's ``run()`` re-raises a captured SIGTERM after shutdown, which would
        # kill the process before the cleanup below (the socket file stayed behind). Windows' event loop
        # cannot install them (NotImplementedError); Ctrl+C / termination end the process there.
        loop = asyncio.get_running_loop()
        for sig in (signal.SIGTERM, signal.SIGINT):
            with contextlib.suppress(NotImplementedError):
                loop.add_signal_handler(sig, lambda: setattr(server, "should_exit", True))
        await server.serve(sockets=listen)

    try:
        asyncio.run(serve())
    finally:
        with contextlib.suppress(Exception):
            engine.unload()
        for leftover in (sock, port_file(sock)):
            with contextlib.suppress(FileNotFoundError):
                leftover.unlink()
    return 0


if __name__ == "__main__":
    sys.exit(main())
