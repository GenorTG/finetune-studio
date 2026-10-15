"""This computer's local-network addresses, to show next to the "local network" switch (Settings, install, CLI).

Shown, never bound one by one: the local network listens on 0.0.0.0 because DHCP moves these addresses.
Container bridges, VPN tunnels and Tailscale are left out (they are not "devices on my Wi-Fi").
"""
from __future__ import annotations

import ipaddress
from dataclasses import dataclass

from finetune_studio.access.tailscale import is_tailnet_ip

VIRTUAL_PREFIXES = ("lo", "docker", "br-", "veth", "virbr", "lxc", "lxd", "cni", "flannel", "vxlan", "tailscale",
                    "tun", "tap", "wg", "zt", "podman", "vmnet", "vboxnet")


@dataclass(frozen=True)
class LanAddress:
    interface: str
    ip: str

    def public(self) -> dict[str, str]:
        return {"interface": self.interface, "ip": self.ip}


def lan_addresses() -> list[LanAddress]:
    try:
        import psutil
    except ImportError:
        return []
    stats = psutil.net_if_stats()
    out: list[LanAddress] = []
    for name, addrs in sorted(psutil.net_if_addrs().items()):
        if name.lower().startswith(VIRTUAL_PREFIXES) or (name in stats and not stats[name].isup):
            continue
        for a in addrs:
            try:
                ip = ipaddress.ip_address(a.address.split("%")[0])
            except ValueError:
                continue                                   # MAC addresses (AF_PACKET) and the like
            if ip.version != 4 or ip.is_loopback or ip.is_link_local or is_tailnet_ip(str(ip)):
                continue
            out.append(LanAddress(interface=name, ip=str(ip)))
    return out
