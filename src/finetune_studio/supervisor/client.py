"""Blocking client for the control socket (stdlib only; used by the CLI and the web app)."""
from __future__ import annotations

import http.client
import json
import socket
from pathlib import Path
from typing import Any

from finetune_studio.supervisor.paths import socket_path


class SupervisorUnavailable(RuntimeError):
    """The control socket is missing or nobody answers on it."""


class SupervisorError(RuntimeError):
    def __init__(self, status: int, message: str) -> None:
        super().__init__(message)
        self.status = status


class _UnixConnection(http.client.HTTPConnection):
    def __init__(self, path: str, timeout: float) -> None:
        super().__init__("localhost", timeout=timeout)
        self._path = path

    def connect(self) -> None:
        self.sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        self.sock.settimeout(self.timeout)
        self.sock.connect(self._path)


class SupervisorClient:
    def __init__(self, path: Path | None = None, timeout: float = 5.0) -> None:
        self.path = path or socket_path()
        self.timeout = timeout

    def _call(self, method: str, url: str) -> dict[str, Any]:
        conn = _UnixConnection(str(self.path), self.timeout)
        try:
            conn.request(method, url)
            resp = conn.getresponse()
            body = resp.read()
        except OSError as exc:
            raise SupervisorUnavailable(f"no supervisor on {self.path}: {exc}") from exc
        finally:
            conn.close()
        try:
            data = json.loads(body or b"{}")
        except ValueError:
            data = {"error": body.decode("utf-8", "replace")[:200]}
        if resp.status >= 400:
            raise SupervisorError(resp.status, str(data.get("error", resp.reason)))
        return data

    def status(self) -> dict[str, Any]:
        return self._call("GET", "/v1/status")

    def events(self, since: int = 0, limit: int = 200) -> list[dict[str, Any]]:
        return self._call("GET", f"/v1/events?since={since}&limit={limit}")["events"]

    def logs(self, name: str, lines: int = 100) -> list[str]:
        return self._call("GET", f"/v1/components/{name}/logs?lines={lines}")["lines"]

    def act(self, name: str, action: str) -> dict[str, Any]:
        return self._call("POST", f"/v1/components/{name}/{action}")

    def shutdown(self) -> dict[str, Any]:
        return self._call("POST", "/v1/shutdown")
