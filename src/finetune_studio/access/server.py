"""The listen address: ``$FTS_ROOT/server.json``. Default 127.0.0.1 — this computer only.

The supervisor (and ``fts webui``) read it at start; command-line ``--host``/``--port`` still override it.
"""
from __future__ import annotations

import ipaddress
import json
from dataclasses import asdict, dataclass

from finetune_studio.access.paths import server_file, write_private

DEFAULT_HOST = "127.0.0.1"
DEFAULT_PORT = 7860


@dataclass(frozen=True)
class ServerAddress:
    host: str = DEFAULT_HOST
    port: int = DEFAULT_PORT
    # Extra host names the browser may use while listening on loopback (e.g. a reverse proxy's public name).
    allowed_hosts: tuple[str, ...] = ()

    @property
    def loopback_only(self) -> bool:
        return is_loopback(self.host)


def is_loopback(host: str) -> bool:
    h = host.strip().strip("[]").lower()
    if h == "localhost":
        return True
    try:
        return ipaddress.ip_address(h).is_loopback
    except ValueError:
        return False                       # a hostname or 0.0.0.0 / :: = reachable from the network


def validate(host: str, port: int, allowed_hosts: tuple[str, ...] | list[str] = ()) -> ServerAddress:
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
    return ServerAddress(host=host, port=int(port), allowed_hosts=names)


def load() -> ServerAddress:
    f = server_file()
    if not f.exists():
        return ServerAddress()
    try:
        data = json.loads(f.read_text(encoding="utf-8"))
        return validate(str(data.get("host", DEFAULT_HOST)), int(data.get("port", DEFAULT_PORT)),
                        list(data.get("allowed_hosts", [])))
    except (OSError, ValueError, TypeError) as e:
        raise ValueError(f"{f} is not a valid server address file: {e}") from e


def resolve(host: str | None, port: int | None) -> ServerAddress:
    """Command-line values win; anything not given comes from server.json (default 127.0.0.1:7860)."""
    saved = load()
    return validate(host or saved.host, port or saved.port, saved.allowed_hosts)


def save(address: ServerAddress) -> None:
    data = asdict(address)
    data["allowed_hosts"] = list(address.allowed_hosts)
    write_private(server_file(), json.dumps(data, indent=2) + "\n")
