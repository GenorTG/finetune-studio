"""The listen address: ``$FTS_ROOT/server.json``. Default 127.0.0.1 — this computer only.

The supervisor (and ``fts webui``) read it at start; command-line ``--host``/``--port`` still override it.
``tailscale: true`` adds a second listener on this computer's Tailscale address (``access.tailscale``), so tailnet
devices reach it while the LAN does not. The last detected address and MagicDNS names are kept, so a start before
tailscaled is up still binds it (IP_FREEBIND) and still accepts those host names.
"""
from __future__ import annotations

import ipaddress
import json
import logging
from dataclasses import asdict, dataclass, replace
from typing import Any

from finetune_studio.access.paths import server_file, write_private

_log = logging.getLogger(__name__)

DEFAULT_HOST = "127.0.0.1"
DEFAULT_PORT = 7860


@dataclass(frozen=True)
class ServerAddress:
    host: str = DEFAULT_HOST
    port: int = DEFAULT_PORT
    # Extra host names the browser may use while listening on loopback (e.g. a reverse proxy's public name).
    allowed_hosts: tuple[str, ...] = ()
    tailscale: bool = False
    tailscale_ip: str = ""
    tailscale_names: tuple[str, ...] = ()

    @property
    def loopback_only(self) -> bool:
        return is_loopback(self.host) and not self.tailscale

    def extra_hosts(self) -> tuple[str, ...]:
        """Addresses to listen on besides ``host``: the Tailscale IP, unless ``host`` already covers it."""
        if not (self.tailscale and self.tailscale_ip) or is_wildcard(self.host) or self.host == self.tailscale_ip:
            return ()
        return (self.tailscale_ip,)


def describe(address: ServerAddress) -> str:
    """Who can reach it, in words (CLI, install output)."""
    if not is_loopback(address.host):
        return "the network"
    if address.tailscale:
        return f"this computer + Tailscale devices ({address.tailscale_ip})"
    return "this computer only"


def is_loopback(host: str) -> bool:
    h = host.strip().strip("[]").lower()
    if h == "localhost":
        return True
    try:
        return ipaddress.ip_address(h).is_loopback
    except ValueError:
        return False                       # a hostname or 0.0.0.0 / :: = reachable from the network


def is_wildcard(host: str) -> bool:
    return host.strip().strip("[]") in ("0.0.0.0", "::", "")


def validate(host: str, port: int, allowed_hosts: tuple[str, ...] | list[str] = (), *, tailscale: bool = False,
             tailscale_ip: str = "", tailscale_names: tuple[str, ...] | list[str] = ()) -> ServerAddress:
    host = host.strip()
    if not host:
        raise ValueError("host is empty")
    if any(c.isspace() for c in host) or "/" in host:
        raise ValueError(f"not a host or IP address: {host!r}")
    if not 1 <= int(port) <= 65535:
        raise ValueError(f"port must be 1-65535, got {port}")
    names = tuple(sorted({n.strip().lower() for n in allowed_hosts if n.strip()}))
    bad = [n for n in names if any(c.isspace() or c in "/:@" for c in n)]
    if bad:
        raise ValueError(f"allowed host names must be bare names, got {bad}")
    from finetune_studio.access.tailscale import is_tailnet_ip
    tailscale_ip = tailscale_ip.strip()
    if tailscale_ip and not is_tailnet_ip(tailscale_ip):
        raise ValueError(f"{tailscale_ip!r} is not a Tailscale address")
    if tailscale and not tailscale_ip:
        raise ValueError("listening on Tailscale needs this computer's Tailscale address: Tailscale is not detected")
    ts_names = tuple(sorted({n.strip().lower() for n in tailscale_names if n.strip()}))
    return ServerAddress(host=host, port=int(port), allowed_hosts=names, tailscale=bool(tailscale),
                         tailscale_ip=tailscale_ip, tailscale_names=ts_names)


def load() -> ServerAddress:
    f = server_file()
    if not f.exists():
        return ServerAddress()
    try:
        data = json.loads(f.read_text(encoding="utf-8"))
        return validate(str(data.get("host", DEFAULT_HOST)), int(data.get("port", DEFAULT_PORT)),
                        list(data.get("allowed_hosts", [])), tailscale=bool(data.get("tailscale", False)),
                        tailscale_ip=str(data.get("tailscale_ip", "")),
                        tailscale_names=list(data.get("tailscale_names", [])))
    except (OSError, ValueError, TypeError) as e:
        raise ValueError(f"{f} is not a valid server address file: {e}") from e


def resolve(host: str | None, port: int | None) -> ServerAddress:
    """Command-line values win; anything not given comes from server.json (default 127.0.0.1:7860)."""
    saved = load()
    given = validate(host or saved.host, port or saved.port)
    return replace(saved, host=given.host, port=given.port)


def with_tailscale(address: ServerAddress, enabled: bool, tailnet: Any = None) -> ServerAddress:
    """``address`` with Tailscale listening on/off; ``tailnet`` (``tailscale.detect()``) refreshes the IP/names."""
    ip, names = address.tailscale_ip, address.tailscale_names
    if tailnet is not None:
        ip, names = tailnet.ipv4, tailnet.host_names()
    return validate(address.host, address.port, address.allowed_hosts, tailscale=enabled, tailscale_ip=ip,
                    tailscale_names=names)


def refresh_tailscale(address: ServerAddress) -> ServerAddress:
    """At service start: follow a changed Tailscale address (saved back); keep the last known one when it is down."""
    if not address.tailscale:
        return address
    from finetune_studio.access import tailscale
    tailnet = tailscale.detect()
    if tailnet is None:
        _log.warning("Tailscale listening is on but Tailscale is not detected; binding the last known address %s",
                     address.tailscale_ip)
        return address
    fresh = with_tailscale(address, True, tailnet)
    if fresh != address:
        save(fresh)
    return fresh


def save(address: ServerAddress) -> None:
    data = asdict(address)
    data["allowed_hosts"] = list(address.allowed_hosts)
    data["tailscale_names"] = list(address.tailscale_names)
    if not (address.tailscale or address.tailscale_ip):          # keep server.json short when Tailscale is unused
        for k in ("tailscale", "tailscale_ip", "tailscale_names"):
            data.pop(k)
    write_private(server_file(), json.dumps(data, indent=2) + "\n")
