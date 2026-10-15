"""Access settings at install time: a KEY=VALUE file for automated installs, or one question on a terminal.

File format (``install.sh --config FILE`` / ``FTS_INSTALL_CONFIG=FILE``), parsed, never sourced::

    FTS_LISTEN_LAN=1                 # devices on the local network / Wi-Fi (this computer always can)
    FTS_TAILSCALE=1                  # your Tailscale devices
    FTS_LISTEN_CUSTOM=198.51.100.2       # other addresses of this computer, comma separated (0.0.0.0 = every network)
    FTS_PORT=7860
    FTS_AUTH=password                # password | off
    FTS_ADMIN_PASSWORD_FILE=/run/secrets/fts   # or FTS_ADMIN_PASSWORD=... (keep the file chmod 600)
    FTS_ALLOWED_HOSTS=studio.example # extra host names (reverse proxy), comma separated
    FTS_INSTALL_SERVICE=1            # install.sh also installs the systemd user unit

Anything beyond this computer needs FTS_AUTH=password. ``FTS_BIND_HOST`` (the older single address: 0.0.0.0 =
local network + Tailscale) still works. Unknown keys are an error (a typo must not silently leave it open).
"""
from __future__ import annotations

import getpass
import shlex
from dataclasses import dataclass
from pathlib import Path

from finetune_studio.access import server, store, tailscale

KEYS = {"FTS_BIND_HOST", "FTS_PORT", "FTS_AUTH", "FTS_ADMIN_PASSWORD", "FTS_ADMIN_PASSWORD_FILE",
        "FTS_ALLOWED_HOSTS", "FTS_INSTALL_SERVICE", "FTS_TAILSCALE", "FTS_LISTEN_LAN", "FTS_LISTEN_CUSTOM"}
YES, NO = ("1", "yes", "true", "on"), ("0", "no", "false", "off")


@dataclass
class InstallAccess:
    host: str | None = None
    port: int | None = None
    auth: str | None = None            # "password" | "off" | None (leave as is)
    password: str | None = None
    allowed_hosts: tuple[str, ...] | None = None
    lan: bool | None = None            # None = leave as is
    tailscale: bool | None = None
    custom: tuple[str, ...] | None = None
    install_service: bool = False


def parse(path: Path) -> InstallAccess:
    values: dict[str, str] = {}
    for n, raw in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        if "=" not in line:
            raise ValueError(f"{path}:{n}: expected KEY=VALUE")
        key, _, value = line.partition("=")
        key = key.strip().removeprefix("export ").strip()
        if key not in KEYS:
            raise ValueError(f"{path}:{n}: unknown key {key!r} (known: {', '.join(sorted(KEYS))})")
        parts = shlex.split(value, comments=True)
        values[key] = parts[0] if parts else ""
    cfg = InstallAccess()
    if values.get("FTS_BIND_HOST"):
        cfg.host = values["FTS_BIND_HOST"]
    if values.get("FTS_PORT"):
        cfg.port = int(values["FTS_PORT"])
    if values.get("FTS_AUTH"):
        if values["FTS_AUTH"] not in ("password", "off"):
            raise ValueError(f"FTS_AUTH must be 'password' or 'off', got {values['FTS_AUTH']!r}")
        cfg.auth = values["FTS_AUTH"]
    if values.get("FTS_ADMIN_PASSWORD_FILE"):
        cfg.password = Path(values["FTS_ADMIN_PASSWORD_FILE"]).read_text(encoding="utf-8").rstrip("\r\n")
    elif values.get("FTS_ADMIN_PASSWORD"):
        cfg.password = values["FTS_ADMIN_PASSWORD"]
    if "FTS_ALLOWED_HOSTS" in values:
        cfg.allowed_hosts = tuple(h for h in values["FTS_ALLOWED_HOSTS"].split(",") if h.strip())
    cfg.install_service = values.get("FTS_INSTALL_SERVICE", "") in ("1", "yes", "true")
    for key, attr in (("FTS_LISTEN_LAN", "lan"), ("FTS_TAILSCALE", "tailscale")):
        if values.get(key):
            if values[key].lower() not in YES + NO:
                raise ValueError(f"{key} must be 1 or 0, got {values[key]!r}")
            setattr(cfg, attr, values[key].lower() in YES)
    if "FTS_LISTEN_CUSTOM" in values:
        cfg.custom = tuple(h for h in values["FTS_LISTEN_CUSTOM"].split(",") if h.strip())
    if cfg.auth == "password" and not cfg.password:
        raise ValueError("FTS_AUTH=password needs FTS_ADMIN_PASSWORD_FILE (or FTS_ADMIN_PASSWORD)")
    return cfg


def apply(cfg: InstallAccess) -> list[str]:
    """Validate everything first, then write. Returns human-readable lines of what changed."""
    current = server.load()
    tailnet = tailscale.detect() if cfg.tailscale else None
    if cfg.tailscale and tailnet is None and not current.tailscale_ip:
        raise ValueError("FTS_TAILSCALE=1 but Tailscale is not detected on this computer (tailscale up first)")
    base = current
    if cfg.host:
        legacy = server.from_host(cfg.host, cfg.port or current.port)
        base = server.change(current, lan=legacy.lan, tailscale=legacy.tailscale, custom=legacy.custom)
    addr = server.change(base, port=cfg.port, lan=cfg.lan, tailscale=cfg.tailscale, custom=cfg.custom,
                         allowed_hosts=cfg.allowed_hosts, tailnet=tailnet)
    password_on = cfg.auth == "password" or (cfg.auth is None and store.load().enabled)
    if not addr.loopback_only and not password_on:
        raise ValueError(f"{server.describe(addr)} can reach this instance but FTS_AUTH is not 'password'. "
                         "Set a password, or keep it to this computer.")
    done: list[str] = []
    if cfg.auth == "password":
        store.set_password(cfg.password or "")
        done.append("password set: sign-in required")
    elif cfg.auth == "off":
        store.disable()
        done.append("password off")
    if addr != current or not server_file_exists():
        server.save(addr)
        done.append(f"reachable from {server.describe(addr)}, port {addr.port}")
    return done


def server_file_exists() -> bool:
    from finetune_studio.access.paths import server_file
    return server_file().exists()


def _yes(question: str) -> bool:
    return input(f"{question} [y/N]: ").strip().lower() in ("y", "yes")


def interactive() -> list[str]:
    """The install-time questions (one per switch). Skipped (by install.sh) when access was configured before."""
    from finetune_studio.access import netinfo
    print("\nWho should be able to open the Finetune Studio web UI? This computer always can "
          "(http://localhost:7860). Anything else needs a password.")
    lan_ips = ", ".join(a.ip for a in netinfo.lan_addresses())
    lan = _yes(f"  Devices on your local network / Wi-Fi{f' (this computer is {lan_ips})' if lan_ips else ''}?")
    tailnet = tailscale.detect()
    ts = False
    if tailnet:
        name = f", {tailnet.dns_name}" if tailnet.dns_name else ""
        ts = _yes(f"  Your Tailscale devices (Tailscale detected: {tailnet.ipv4}{name})?")
    cfg = InstallAccess(lan=lan, tailscale=ts, auth="password" if lan or ts else None)
    if not (lan or ts):
        return apply(cfg)
    return _with_password(cfg)


def _with_password(cfg: InstallAccess) -> list[str]:
    while True:
        pw = getpass.getpass("Password for the web UI (10+ characters): ")
        if getpass.getpass("Repeat it: ") != pw:
            print("The two entries differ, again.")
            continue
        try:
            cfg.password = pw
            return apply(cfg)
        except ValueError as e:
            print(f"{e} — again.")
