"""The web UI as a managed component: the supervisor owns the listening socket and hands it over."""
from __future__ import annotations

import asyncio
import logging
import socket
import sys
import urllib.request

from finetune_studio.supervisor.spec import ComponentSpec

WEB_APP = "finetune_studio.webui.app:app"
# Open SSE streams (activity feed, jobs) never end by themselves: without a cap, uvicorn waits for them until the
# supervisor's SIGKILL (15 s). In-flight jobs die with the process anyway, so a short cap only makes restarts fast.
GRACEFUL_SHUTDOWN_S = 3
# Linux: bind an address this machine does not hold yet (tailscaled may come up after the service starts).
IP_FREEBIND = getattr(socket, "IP_FREEBIND", 15)

_log = logging.getLogger(__name__)


class ListenSocket:
    """A bound, listening TCP socket that outlives web-process restarts.

    The kernel keeps accepting into the backlog while the web child restarts, so a restart
    or crash shows up as a short wait, not "connection refused". ``release`` closes it when
    the component is stopped or failed for good, so clients then get an honest refusal.
    """

    def __init__(self, host: str, port: int, *, freebind: bool = False) -> None:
        self.host, self.port, self.freebind = host, port, freebind
        self._sock: socket.socket | None = None

    def open(self) -> socket.socket:
        if self._sock is None:
            family = socket.AF_INET6 if ":" in self.host else socket.AF_INET
            sock = socket.socket(family, socket.SOCK_STREAM)
            try:
                sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
                if self.freebind and family == socket.AF_INET and sys.platform.startswith("linux"):
                    sock.setsockopt(socket.IPPROTO_IP, IP_FREEBIND, 1)
                sock.bind((self.host, self.port))
                sock.listen(128)
            except OSError:
                sock.close()
                raise
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


class ExtraListeners:
    """Optional extra addresses (the Tailscale IP). One that cannot be bound is logged and skipped: the web UI
    stays up on the main address instead of failing to start."""

    def __init__(self, hosts: tuple[str, ...], port: int) -> None:
        self.sockets = [ListenSocket(h, port, freebind=True) for h in hosts]
        self.failed: dict[str, str] = {}

    def open(self) -> None:
        for ls in self.sockets:
            try:
                ls.open()
                self.failed.pop(ls.host, None)
            except OSError as e:
                if ls.host not in self.failed:
                    _log.warning("cannot listen on %s:%s too: %s", ls.host, ls.port, e)
                self.failed[ls.host] = str(e)

    def opened(self) -> list[ListenSocket]:
        return [ls for ls in self.sockets if ls.host not in self.failed]

    def close(self) -> None:
        for ls in self.sockets:
            ls.close()


def web_spec(host: str, port: int, *, cwd: str | None = None, extra_hosts: tuple[str, ...] = ()) -> ComponentSpec:
    listen = ListenSocket(host, port)
    extra = ExtraListeners(extra_hosts, port)
    probe_host = "127.0.0.1" if host in ("0.0.0.0", "") else ("::1" if host == "::" else host)
    probe_url = f"http://[{probe_host}]:{port}/api/health" if ":" in probe_host else f"http://{probe_host}:{port}/api/health"

    def fds() -> tuple[int, ...]:
        return (listen.fileno(), *(ls.fileno() for ls in extra.opened()))

    def argv() -> list[str]:
        if not extra.opened():
            return [sys.executable, "-m", "uvicorn", WEB_APP, "--fd", str(listen.fileno()), "--no-access-log",
                    "--timeout-graceful-shutdown", str(GRACEFUL_SHUTDOWN_S)]
        # uvicorn's CLI takes one --fd; webui.serve hands uvicorn several sockets.
        return [sys.executable, "-m", "finetune_studio.webui.serve", *(a for fd in fds() for a in ("--fd", str(fd))),
                "--timeout-graceful-shutdown", str(GRACEFUL_SHUTDOWN_S)]

    def prepare() -> None:
        listen.open()
        extra.open()

    def release() -> None:
        listen.close()
        extra.close()

    def env() -> dict[str, str]:
        # The app reads these to know who can reach it (no-password-on-the-network warning, host checks).
        return {"FTS_LISTEN_HOST": host, "FTS_LISTEN_PORT": str(port),
                "FTS_LISTEN_EXTRA": ",".join(ls.host for ls in extra.opened())}

    return ComponentSpec(
        name="web",
        argv=argv,
        pass_fds=fds,
        prepare=prepare,
        release=release,
        probe=http_probe(probe_url),
        cwd=cwd,
        env=env,
    )
