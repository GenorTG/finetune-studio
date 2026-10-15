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


class UnixHTTPConnection(http.client.HTTPConnection):
    """``http.client`` over an AF_UNIX socket (control socket, inference worker)."""

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

    def _call(self, method: str, url: str, body: dict[str, Any] | None = None) -> dict[str, Any]:
        conn = UnixHTTPConnection(str(self.path), self.timeout)
        try:
            if body is None:
                conn.request(method, url)
            else:
                conn.request(method, url, body=json.dumps(body), headers={"Content-Type": "application/json"})
            resp = conn.getresponse()
            body = resp.read()
        except OSError as exc:
            raise SupervisorUnavailable(f"no supervisor on {self.path}: {exc}") from exc
        finally:
            conn.close()
        try:
            data = json.loads(body or b"{}")
        except ValueError:
            data = {"detail": body.decode("utf-8", "replace")[:200]}
        if resp.status >= 400:
            raise SupervisorError(resp.status, str(data.get("detail", resp.reason)))
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

    # ── jobs ──
    def jobs(self) -> list[dict[str, Any]]:
        return self._call("GET", "/v1/jobs")["jobs"]

    def start_job(self, job_id: str) -> dict[str, Any]:
        return self._call("POST", "/v1/jobs", {"id": job_id})

    def job(self, job_id: str) -> dict[str, Any]:
        return self._call("GET", f"/v1/jobs/{job_id}")

    def stop_job(self, job_id: str) -> dict[str, Any]:
        return self._call("POST", f"/v1/jobs/{job_id}/stop")

    def job_logs(self, job_id: str, lines: int = 100) -> list[str]:
        return self._call("GET", f"/v1/jobs/{job_id}/logs?lines={lines}")["lines"]
