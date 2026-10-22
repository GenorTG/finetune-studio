"""Blocking client for the control socket (stdlib only; used by the CLI and the web app)."""
from __future__ import annotations

import http.client
import json
import os
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


# Windows CPython has no AF_UNIX: there a local server addressed by a socket PATH listens on a loopback port instead,
# written with a per-start token to ``<path>.port`` (the web app died at start-up on Windows; VM 2026-10-22).
HAS_AF_UNIX = hasattr(socket, "AF_UNIX") and os.environ.get("FTS_NO_AF_UNIX") != "1"   # env: test seam for Linux CI
TOKEN_HEADER = "X-FTS-Local-Token"


def port_file(path: str | Path) -> Path:
    return Path(str(path) + ".port")


def read_port_file(path: str | Path) -> tuple[int, str]:
    """(port, token) of the loopback server for ``path``; OSError when it is not running."""
    port, token = port_file(path).read_text(encoding="utf-8").split()
    return int(port), token


def bind_loopback(path: str | Path) -> tuple[socket.socket, str]:
    """Listening loopback socket for the server addressed by ``path``; its port + a new token go to ``<path>.port``."""
    import secrets

    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.bind(("127.0.0.1", 0))
    sock.listen(64)
    token = secrets.token_hex(16)
    tmp = port_file(path).with_suffix(".port.tmp")
    tmp.write_text(f"{sock.getsockname()[1]} {token}", encoding="utf-8")
    os.replace(tmp, port_file(path))
    return sock, token


def require_token(app: Any, token: str) -> Any:
    """ASGI wrapper: a loopback port is reachable by every local process, so each request must carry the token."""
    want = token.encode()

    async def guarded(scope: dict[str, Any], receive: Any, send: Any) -> None:
        if scope["type"] == "http" and dict(scope.get("headers") or []).get(TOKEN_HEADER.lower().encode()) != want:
            await send({"type": "http.response.start", "status": 403, "headers": [(b"content-type", b"text/plain")]})
            await send({"type": "http.response.body", "body": b"missing or wrong local token"})
            return
        await app(scope, receive, send)

    return guarded


class UnixHTTPConnection(http.client.HTTPConnection):
    """``http.client`` to a local server addressed by a socket path (control socket, inference worker).

    AF_UNIX where the platform has it; otherwise the loopback port + token from ``<path>.port``.
    """

    def __init__(self, path: str, timeout: float) -> None:
        super().__init__("localhost", timeout=timeout)
        self._path = path
        self._token = ""

    def connect(self) -> None:
        if HAS_AF_UNIX:
            self.sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
            self.sock.settimeout(self.timeout)
            self.sock.connect(self._path)
            return
        port, _ = read_port_file(self._path)          # missing file: OSError, i.e. "not running"
        self.sock = socket.create_connection(("127.0.0.1", port), self.timeout)

    def putrequest(self, method: str, url: str, *args: Any, **kwargs: Any) -> None:
        super().putrequest(method, url, *args, **kwargs)
        if not HAS_AF_UNIX:
            try:
                self._token = read_port_file(self._path)[1]
            except (OSError, ValueError):
                return                                 # connect() reports the missing server
            self.putheader(TOKEN_HEADER, self._token)


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
