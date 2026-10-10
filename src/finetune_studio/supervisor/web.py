"""The web UI as a managed component: the supervisor owns the listening socket and hands it over."""
from __future__ import annotations

import asyncio
import socket
import sys
import urllib.request

from finetune_studio.supervisor.spec import ComponentSpec

WEB_APP = "finetune_studio.webui.app:app"


class ListenSocket:
    """A bound, listening TCP socket that outlives web-process restarts.

    The kernel keeps accepting into the backlog while the web child restarts, so a restart
    or crash shows up as a short wait, not "connection refused". ``release`` closes it when
    the component is stopped or failed for good, so clients then get an honest refusal.
    """

    def __init__(self, host: str, port: int) -> None:
        self.host, self.port = host, port
        self._sock: socket.socket | None = None

    def open(self) -> socket.socket:
        if self._sock is None:
            family = socket.AF_INET6 if ":" in self.host else socket.AF_INET
            sock = socket.socket(family, socket.SOCK_STREAM)
            sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            sock.bind((self.host, self.port))
            sock.listen(128)
            sock.set_inheritable(True)
            self._sock = sock
        return self._sock

    def close(self) -> None:
        if self._sock is not None:
            self._sock.close()
            self._sock = None

    def fileno(self) -> int:
        return self.open().fileno()


def http_probe(url: str, timeout: float = 4.0):
    """Probe that GETs ``url``; healthy = HTTP 200."""
    def _get() -> tuple[bool, str]:
        try:
            with urllib.request.urlopen(url, timeout=timeout) as resp:
                return resp.status == 200, f"HTTP {resp.status}"
        except Exception as exc:  # noqa: BLE001 - any failure is "unhealthy" with the reason
            return False, f"{type(exc).__name__}: {exc}"[:200]

    async def probe() -> tuple[bool, str]:
        return await asyncio.to_thread(_get)

    return probe


def web_spec(host: str, port: int, *, cwd: str | None = None) -> ComponentSpec:
    listen = ListenSocket(host, port)
    probe_host = "127.0.0.1" if host in ("0.0.0.0", "") else ("::1" if host == "::" else host)
    probe_url = f"http://[{probe_host}]:{port}/api/health" if ":" in probe_host else f"http://{probe_host}:{port}/api/health"
    return ComponentSpec(
        name="web",
        argv=lambda: [sys.executable, "-m", "uvicorn", WEB_APP, "--fd", str(listen.fileno()), "--no-access-log"],
        pass_fds=lambda: (listen.fileno(),),
        prepare=listen.open,
        release=listen.close,
        probe=http_probe(probe_url),
        cwd=cwd,
    )
