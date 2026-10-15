"""The web UI as a managed component: the supervisor owns the listening socket and hands it over."""
from __future__ import annotations

import asyncio
import logging
import socket
import sys
import urllib.request
from collections.abc import Callable

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
                if family == socket.AF_INET6 and self.host.strip("[]") == "::":
                    sock.setsockopt(socket.IPPROTO_IPV6, socket.IPV6_V6ONLY, 0)     # "::" = IPv4 too
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


class Listeners:
    """The web UI's listening sockets for one plan (addresses + port). The first address must bind; any other
    that cannot (a Tailscale IP while tailscaled is down on a non-Linux box) is logged and skipped, so the UI stays
    up. ``open(plan)`` with a different plan closes the old sockets and binds the new ones (a changed setting
    applies on the next web start)."""

    def __init__(self) -> None:
        self.plan: tuple[tuple[str, ...], int] | None = None
        self.sockets: list[ListenSocket] = []
        self.failed: dict[str, str] = {}
        self.error = ""          # why the last requested plan was not used (shown in Settings)

    def open(self, plan: tuple[tuple[str, ...], int]) -> None:
        if plan == self.plan:
            self.error = ""                          # what is saved is what is bound
        else:
            hosts, port = plan
            fresh = [ListenSocket(h, port, freebind=i > 0) for i, h in enumerate(hosts)]
            if self.plan is not None:
                # A busy port or a gone address must not leave the UI unreachable (it is where you fix it).
                self.close()
                try:
                    fresh[0].open()
                except OSError as e:
                    self.error = f"cannot listen on {hosts[0]}:{port}: {e}"
                    _log.error("%s; keeping %s", self.error, ", ".join(f"{h}:{self.plan[1]}" for h in self.plan[0]))
                    self.sockets[0].open()
                    self._bind_extras()
                    return
            self.sockets, self.plan, self.failed, self.error = fresh, plan, {}, ""
            _log.info("web UI listens on %s", ", ".join(f"{h}:{port}" for h in hosts))
        self.sockets[0].open()                       # the main address: failing here is a failed start
        self._bind_extras()

    def _bind_extras(self) -> None:
        for ls in self.sockets[1:]:
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


def _probe_url(host: str, port: int) -> str:
    h = "127.0.0.1" if host in ("0.0.0.0", "") else ("::1" if host == "::" else host)
    return f"http://[{h}]:{port}/api/health" if ":" in h else f"http://{h}:{port}/api/health"


def web_spec(host: str | None = None, port: int | None = None, *, cwd: str | None = None,
             extra_hosts: tuple[str, ...] = (),
             plan: Callable[[], tuple[tuple[str, ...], int]] | None = None) -> ComponentSpec:
    """``plan`` (re-read before every web start) or a fixed ``host`` + ``extra_hosts`` on ``port``."""
    if plan is None:
        fixed = ((host or "127.0.0.1", *extra_hosts), int(port or 0))
        plan = lambda: fixed
    listeners = Listeners()

    def fds() -> tuple[int, ...]:
        return tuple(ls.fileno() for ls in listeners.opened())

    def argv() -> list[str]:
        if len(listeners.opened()) == 1:
            return [sys.executable, "-m", "uvicorn", WEB_APP, "--fd", str(fds()[0]), "--no-access-log",
                    "--timeout-graceful-shutdown", str(GRACEFUL_SHUTDOWN_S)]
        # uvicorn's CLI takes one --fd; webui.serve hands uvicorn several sockets.
        return [sys.executable, "-m", "finetune_studio.webui.serve", *(a for fd in fds() for a in ("--fd", str(fd))),
                "--timeout-graceful-shutdown", str(GRACEFUL_SHUTDOWN_S)]

    def prepare() -> None:
        listeners.open(plan())

    def env() -> dict[str, str]:
        # The app reads these to know who can reach it (no-password warning, Settings -> Access & network).
        bound = [ls.host for ls in listeners.opened()]
        return {"FTS_LISTEN_ADDRS": ",".join(bound), "FTS_LISTEN_HOST": bound[0],
                "FTS_LISTEN_PORT": str(listeners.plan[1] if listeners.plan else port),
                "FTS_LISTEN_ERROR": listeners.error}

    async def probe() -> tuple[bool, str]:
        hosts, p = listeners.plan or plan()
        return await http_probe(_probe_url(hosts[0], p))()

    return ComponentSpec(
        name="web",
        argv=argv,
        pass_fds=fds,
        prepare=prepare,
        release=listeners.close,
        probe=probe,
        cwd=cwd,
        env=env,
    )
