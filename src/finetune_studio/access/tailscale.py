"""Is this computer on a Tailscale network? Used to offer "also listen on Tailscale" (install, Settings, CLI).

Asks the ``tailscale`` CLI first (``tailscale status --json``: IPs + MagicDNS name); without it, looks for a network
interface holding an address in Tailscale's ranges (``TAILNET_V4`` / ``TAILNET_V6``).
"""
from __future__ import annotations

import ipaddress
import json
import logging
import shutil
import subprocess
import time
from dataclasses import dataclass

_log = logging.getLogger(__name__)

TAILNET_V4 = ipaddress.ip_network((0x64400000, 10))          # 100.64/10 (CGNAT), written as a number
TAILNET_V6 = ipaddress.ip_network("fd7a:115c:a1e0::/48")


@dataclass(frozen=True)
class Tailnet:
    ipv4: str
    dns_name: str = ""          # MagicDNS name, e.g. box.tail1234.ts.net ("" when MagicDNS is off or unknown)
    hostname: str = ""
    source: str = "cli"         # "cli" | "interface"

    def host_names(self) -> tuple[str, ...]:
        """Every name a browser on the tailnet may put in the Host header."""
        names = {self.ipv4, self.dns_name, self.dns_name.split(".")[0] if self.dns_name else "", self.hostname}
        return tuple(sorted(n.lower() for n in names if n))

    def public(self) -> dict[str, str]:
        return {"ip": self.ipv4, "dns_name": self.dns_name, "hostname": self.hostname, "source": self.source}


def is_tailnet_ip(value: str) -> bool:
    try:
        ip = ipaddress.ip_address(value.strip().strip("[]"))
    except ValueError:
        return False
    return ip in (TAILNET_V4 if ip.version == 4 else TAILNET_V6)


def _from_cli(timeout: float) -> Tailnet | None:
    exe = shutil.which("tailscale")
    if not exe:
        return None
    try:
        out = subprocess.run([exe, "status", "--json"], capture_output=True, text=True, timeout=timeout, check=False)
    except (OSError, subprocess.TimeoutExpired) as e:
        _log.info("tailscale status failed: %s", e)
        return None
    if out.returncode != 0 or not out.stdout.strip():
        return None
    return parse_status(out.stdout)


def parse_status(text: str) -> Tailnet | None:
    """``tailscale status --json`` -> Tailnet, or None when tailscaled is not connected."""
    try:
        data = json.loads(text)
    except ValueError:
        return None
    me = data.get("Self") or {}
    if data.get("BackendState") not in (None, "Running"):
        return None
    v4 = next((ip for ip in me.get("TailscaleIPs") or [] if ":" not in ip and is_tailnet_ip(ip)), "")
    if not v4:
        return None
    return Tailnet(ipv4=v4, dns_name=str(me.get("DNSName") or "").rstrip(".").lower(),
                   hostname=str(me.get("HostName") or "").lower(), source="cli")


def _from_interfaces() -> Tailnet | None:
    try:
        import psutil
    except ImportError:
        return None
    for name, addrs in psutil.net_if_addrs().items():
        for a in addrs:
            if ":" not in a.address and is_tailnet_ip(a.address):
                _log.info("Tailscale address %s found on interface %s (no tailscale CLI)", a.address, name)
                return Tailnet(ipv4=a.address, source="interface")
    return None


def detect(timeout: float = 3.0) -> Tailnet | None:
    """This computer's Tailscale identity, or None (not installed, not logged in, or down).

    With the CLI installed its answer is final (a stale tailscale0 address while logged out is not "connected")."""
    if shutil.which("tailscale"):
        return _from_cli(timeout)
    return _from_interfaces()


_cached: dict[str, object] = {"at": None, "value": None}


def detect_cached(max_age_s: float = 30.0) -> Tailnet | None:
    """``detect()``, reused for ``max_age_s`` (the Settings page polls; ``tailscale status`` is a subprocess)."""
    at = _cached["at"]
    if at is None or time.monotonic() - float(at) > max_age_s:  # type: ignore[arg-type]
        _cached["value"], _cached["at"] = detect(), time.monotonic()
    return _cached["value"]  # type: ignore[return-value]
