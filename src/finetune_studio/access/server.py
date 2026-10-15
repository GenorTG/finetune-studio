"""Who can reach the web UI: ``$FTS_ROOT/server.json``. Default: this computer only (127.0.0.1:7860).

Switches instead of one address (Genor 2026-10-15):

- **this computer**: always on (the service's own health check and the CLI use it);
- **local network** (``lan``): devices on your LAN / Wi-Fi;
- **Tailscale** (``tailscale``): your tailnet devices (``access.tailscale``);
- **custom** addresses: anything else (``198.51.100.2``, ``::``, ``0.0.0.0``).

``binds()`` is as narrow as the switches allow: 127.0.0.1 + the Tailscale IP + custom addresses, or 0.0.0.0 once the
local network is on (LAN addresses change with DHCP). Whatever is bound, ``allows()`` decides per connection by the
address it arrived on (``access.middleware``), so switching something off works at once; switching something on or a
new port needs the web UI to restart (the supervisor re-reads this file on every web start).

Files written before 2026-10-15 hold one ``host``: 127.0.0.1 = this computer, 0.0.0.0 = local network + Tailscale,
any other address = a custom address. ``--host`` on the command line is read the same way.
"""
from __future__ import annotations

import ipaddress
import json
import logging
from dataclasses import dataclass, replace
from typing import Any

from finetune_studio.access.paths import server_file, write_private

_log = logging.getLogger(__name__)

DEFAULT_HOST = "127.0.0.1"
DEFAULT_PORT = 7860
WILDCARDS = ("0.0.0.0", "::")


def _norm(host: str) -> str:
    return host.strip().strip("[]").lower()


def is_loopback(host: str) -> bool:
    h = _norm(host)
    if h == "localhost":
        return True
    try:
        return ipaddress.ip_address(h).is_loopback
    except ValueError:
        return False                       # a hostname or 0.0.0.0 / :: = reachable from the network


def is_wildcard(host: str) -> bool:
    return _norm(host) in (*WILDCARDS, "")


@dataclass(frozen=True)
class ServerAddress:
    port: int = DEFAULT_PORT
    lan: bool = False
    tailscale: bool = False
    custom: tuple[str, ...] = ()
    # Extra host names the browser may use while the local network is off (e.g. a reverse proxy's public name).
    allowed_hosts: tuple[str, ...] = ()
    # Last detected Tailscale address and MagicDNS names: bound (IP_FREEBIND) even while tailscaled is still down.
    tailscale_ip: str = ""
    tailscale_names: tuple[str, ...] = ()

    @property
    def loopback_only(self) -> bool:
        return not (self.lan or self.tailscale or any(not is_loopback(c) for c in self.custom))

    @property
    def everything(self) -> bool:
        """A custom 0.0.0.0 / :: = every address, no per-connection filter."""
        return any(is_wildcard(c) for c in self.custom)

    def binds(self) -> tuple[str, ...]:
        """Addresses to open listening sockets on; the first one must work (the others are skipped on failure)."""
        if "::" in self.custom:
            return ("::",)                                  # dual stack: IPv4 too
        if self.lan or self.everything:
            return ("0.0.0.0", *(c for c in self.custom if ":" in c))
        out = [DEFAULT_HOST]
        if self.tailscale and self.tailscale_ip:
            out.append(self.tailscale_ip)
        out += [c for c in self.custom if c not in out]
        return tuple(out)

    @property
    def host(self) -> str:
        """The main bind address (what older code and messages call "the" address)."""
        return self.binds()[0]

    def probe_host(self) -> str:
        """Where this machine itself reaches the web UI (health checks, ``fts`` hints)."""
        return DEFAULT_HOST if self.host in ("0.0.0.0", DEFAULT_HOST) else ("::1" if self.host == "::" else self.host)

    def allows(self, local_ip: str) -> bool:
        """May a connection that arrived on ``local_ip`` (this machine's address) use the web UI?"""
        from finetune_studio.access.tailscale import is_tailnet_ip
        ip = _norm(local_ip).removeprefix("::ffff:")       # IPv4 arriving on a dual-stack socket
        if is_loopback(ip) or self.everything or ip in self.custom:
            return True
        if is_tailnet_ip(ip):
            return self.tailscale
        return self.lan

    @staticmethod
    def kind_of(local_ip: str) -> str:
        from finetune_studio.access.tailscale import is_tailnet_ip
        ip = _norm(local_ip).removeprefix("::ffff:")
        if is_loopback(ip):
            return "this computer"
        return "Tailscale" if is_tailnet_ip(ip) else "the local network"

    def host_names(self) -> set[str] | None:
        """Host header names a browser may use; None = any (the local network is on: LAN names vary)."""
        if self.lan or self.everything:
            return None
        names = {"localhost", "127.0.0.1", "::1", "[::1]", *self.allowed_hosts, *self.custom}
        if self.tailscale:
            names |= {self.tailscale_ip, *self.tailscale_names}
        return {n for n in names if n}


def describe(address: ServerAddress) -> str:
    """Who can reach it, in words (CLI, install output). No addresses: install-service.sh parses the line."""
    if address.everything:
        return "every network"
    parts = ["this computer"]
    if address.lan:
        parts.append("local network")
    if address.tailscale:
        parts.append("Tailscale")
    custom = [c for c in address.custom if not is_loopback(c)]
    if custom:
        parts.append(f"{len(custom)} custom address{'es' if len(custom) > 1 else ''}")
    return " + ".join(parts) + (" only" if len(parts) == 1 else "")


def make(port: int = DEFAULT_PORT, *, lan: bool = False, tailscale: bool = False,
         custom: tuple[str, ...] | list[str] = (), allowed_hosts: tuple[str, ...] | list[str] = (),
         tailscale_ip: str = "", tailscale_names: tuple[str, ...] | list[str] = ()) -> ServerAddress:
    """Validated ServerAddress (ValueError with the reason otherwise)."""
    from finetune_studio.access.tailscale import is_tailnet_ip
    if not 1 <= int(port) <= 65535:
        raise ValueError(f"port must be 1-65535, got {port}")
    addrs: list[str] = []
    for raw in custom:
        c = _norm(raw)
        if not c:
            continue
        try:
            ipaddress.ip_address(c)
        except ValueError:
            raise ValueError(f"custom listen addresses must be IP addresses of this computer, got {raw!r}") from None
        if c not in addrs:
            addrs.append(c)
    names = tuple(sorted({n.strip().lower() for n in allowed_hosts if n.strip()}))
    bad = [n for n in names if any(c.isspace() or c in "/:@" for c in n)]
    if bad:
        raise ValueError(f"allowed host names must be bare names, got {bad}")
    tailscale_ip = tailscale_ip.strip()
    if tailscale_ip and not is_tailnet_ip(tailscale_ip):
        raise ValueError(f"{tailscale_ip!r} is not a Tailscale address")
    if tailscale and not tailscale_ip and not lan:
        raise ValueError("listening on Tailscale needs this computer's Tailscale address: Tailscale is not detected")
    ts_names = tuple(sorted({n.strip().lower() for n in tailscale_names if n.strip()}))
    return ServerAddress(port=int(port), lan=bool(lan), tailscale=bool(tailscale), custom=tuple(addrs),
                         allowed_hosts=names, tailscale_ip=tailscale_ip, tailscale_names=ts_names)


def from_host(host: str, port: int = DEFAULT_PORT, allowed_hosts: tuple[str, ...] | list[str] = ()) -> ServerAddress:
    """The pre-switches single address: 127.0.0.1 = this computer, 0.0.0.0 = LAN + Tailscale, else custom."""
    h = _norm(host)
    if not h:
        raise ValueError("host is empty")
    if any(c.isspace() for c in h) or "/" in h:
        raise ValueError(f"not a host or IP address: {host!r}")
    if is_loopback(h):
        return make(port, allowed_hosts=allowed_hosts)
    if h == "0.0.0.0":
        return make(port, lan=True, tailscale=True, allowed_hosts=allowed_hosts)
    return make(port, custom=[h], allowed_hosts=allowed_hosts)


# Kept for callers that think in one address (old tests, scripts).
validate = from_host


def load() -> ServerAddress:
    f = server_file()
    if not f.exists():
        return ServerAddress()
    try:
        data = json.loads(f.read_text(encoding="utf-8"))
        port = int(data.get("port", DEFAULT_PORT))
        allowed = list(data.get("allowed_hosts", []))
        if "lan" not in data and "custom" not in data:              # written before the switches
            base = from_host(str(data.get("host", DEFAULT_HOST)), port, allowed)
            if data.get("tailscale") or data.get("tailscale_ip"):
                base = replace(base, tailscale=base.tailscale or bool(data.get("tailscale")),
                               tailscale_ip=str(data.get("tailscale_ip", "")),
                               tailscale_names=tuple(data.get("tailscale_names", [])))
            return base
        return make(port, lan=bool(data.get("lan")), tailscale=bool(data.get("tailscale")),
                    custom=list(data.get("custom", [])), allowed_hosts=allowed,
                    tailscale_ip=str(data.get("tailscale_ip", "")),
                    tailscale_names=list(data.get("tailscale_names", [])))
    except (OSError, ValueError, TypeError) as e:
        raise ValueError(f"{f} is not a valid server address file: {e}") from e


def resolve(host: str | None, port: int | None) -> ServerAddress:
    """Command-line values win (``--host`` read as a pre-switches address); the rest comes from server.json."""
    saved = load()
    if host:
        given = from_host(host, port or saved.port, saved.allowed_hosts)
        return replace(given, tailscale_ip=saved.tailscale_ip, tailscale_names=saved.tailscale_names)
    return replace(saved, port=make(port or saved.port).port)


def change(address: ServerAddress, *, port: int | None = None, lan: bool | None = None,
           tailscale: bool | None = None, custom: list[str] | tuple[str, ...] | None = None,
           allowed_hosts: list[str] | tuple[str, ...] | None = None, tailnet: Any = None) -> ServerAddress:
    """A validated copy with the given switches changed (None = keep); ``tailnet`` refreshes the Tailscale IP."""
    ip, names = address.tailscale_ip, address.tailscale_names
    if tailnet is not None:
        ip, names = tailnet.ipv4, tailnet.host_names()
    return make(address.port if port is None else port,
                lan=address.lan if lan is None else lan,
                tailscale=address.tailscale if tailscale is None else tailscale,
                custom=address.custom if custom is None else custom,
                allowed_hosts=address.allowed_hosts if allowed_hosts is None else allowed_hosts,
                tailscale_ip=ip, tailscale_names=names)


def with_tailscale(address: ServerAddress, enabled: bool, tailnet: Any = None) -> ServerAddress:
    return change(address, tailscale=enabled, tailnet=tailnet)


def refresh_tailscale(address: ServerAddress) -> ServerAddress:
    """At web start: follow a changed Tailscale address (saved back); keep the last known one while it is down."""
    if not address.tailscale:
        return address
    from finetune_studio.access import tailscale
    tailnet = tailscale.detect()
    if tailnet is None:
        _log.warning("Tailscale is switched on but not detected; keeping the last known address %s",
                     address.tailscale_ip or "(none)")
        return address
    fresh = with_tailscale(address, True, tailnet)
    if fresh != address:
        try:
            save(fresh)
        except OSError as e:
            _log.warning("could not save the new Tailscale address: %s", e)
    return fresh


def save(address: ServerAddress) -> None:
    data: dict[str, Any] = {"port": address.port, "lan": address.lan, "tailscale": address.tailscale,
                            "custom": list(address.custom), "allowed_hosts": list(address.allowed_hosts)}
    if address.tailscale_ip:
        data["tailscale_ip"] = address.tailscale_ip
        data["tailscale_names"] = list(address.tailscale_names)
    write_private(server_file(), json.dumps(data, indent=2) + "\n")


def effective(bound: tuple[str, ...], saved: ServerAddress) -> list[str]:
    """Who can reach it right now: what is bound, filtered by the saved switches (``allows``)."""
    wildcard = any(is_wildcard(b) for b in bound)
    if saved.everything and wildcard:
        return ["every network"]
    kinds = ["this computer"]
    if saved.lan and wildcard:
        kinds.append("local network")
    if saved.tailscale and (wildcard or (saved.tailscale_ip and saved.tailscale_ip in bound)):
        kinds.append("Tailscale")
    kinds += [c for c in saved.custom if not is_loopback(c) and (c in bound or wildcard)]
    return kinds
