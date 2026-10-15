"""One line per HTTP request in ``logs/web-access.log``; a request id on every log line made while serving it.

Line: ``<method> <path>[?<query, secrets redacted>] <status> <ms>ms <client>``. The id is returned as
``X-Request-ID`` so an error the user reports can be found in ``web.log`` and ``web-access.log``.
"""
from __future__ import annotations

import secrets
import time
from typing import Any

from finetune_studio import logsetup


class AccessLogMiddleware:
    def __init__(self, app: Any) -> None:
        self.app = app
        self.log = logsetup.access_logger()

    async def __call__(self, scope: dict[str, Any], receive: Any, send: Any) -> None:
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return
        rid = secrets.token_hex(6)
        token = logsetup.request_id.set(rid)
        started = time.perf_counter()
        status = 500

        async def send_wrapper(message: dict[str, Any]) -> None:
            nonlocal status
            if message["type"] == "http.response.start":
                status = message["status"]
                message.setdefault("headers", [])
                message["headers"] = [*message["headers"], (b"x-request-id", rid.encode())]
            await send(message)

        try:
            await self.app(scope, receive, send_wrapper)
        finally:
            qs = scope.get("query_string", b"").decode("latin-1")
            target = scope.get("path", "") + (f"?{qs}" if qs else "")
            client = (scope.get("client") or ("-", 0))[0]
            ms = (time.perf_counter() - started) * 1000
            self.log.info("%s %s %d %.0fms %s", scope.get("method", "-"), target, status, ms, client)
            logsetup.request_id.reset(token)
