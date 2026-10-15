"""Access settings at install time: a KEY=VALUE file for automated installs, or one question on a terminal.

File format (``install.sh --config FILE`` / ``FTS_INSTALL_CONFIG=FILE``), parsed, never sourced::

    FTS_BIND_HOST=0.0.0.0            # 127.0.0.1 (default) = this computer only
    FTS_PORT=7860
    FTS_AUTH=password                # password | off
    FTS_ADMIN_PASSWORD_FILE=/run/secrets/fts   # or FTS_ADMIN_PASSWORD=... (keep the file chmod 600)
    FTS_ALLOWED_HOSTS=studio.example # extra host names (reverse proxy), comma separated
    FTS_TAILSCALE=1                  # also listen on this computer's Tailscale address (needs FTS_AUTH=password)
    FTS_INSTALL_SERVICE=1            # install.sh also installs the systemd user unit

Unknown keys are an error (a typo must not silently leave the instance open).
"""
from __future__ import annotations

import getpass
import shlex
from dataclasses import dataclass
from pathlib import Path

from finetune_studio.access import server, store, tailscale

KEYS = {"FTS_BIND_HOST", "FTS_PORT", "FTS_AUTH", "FTS_ADMIN_PASSWORD", "FTS_ADMIN_PASSWORD_FILE",
        "FTS_ALLOWED_HOSTS", "FTS_INSTALL_SERVICE", "FTS_TAILSCALE"}


@dataclass
class InstallAccess:
    host: str | None = None
    port: int | None = None
    auth: str | None = None            # "password" | "off" | None (leave as is)
    password: str | None = None
    allowed_hosts: tuple[str, ...] | None = None
    tailscale: bool | None = None      # None = leave as is
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
    if values.get("FTS_TAILSCALE"):
        if values["FTS_TAILSCALE"] not in ("0", "1", "yes", "no", "true", "false"):
            raise ValueError(f"FTS_TAILSCALE must be 1 or 0, got {values['FTS_TAILSCALE']!r}")
        cfg.tailscale = values["FTS_TAILSCALE"] in ("1", "yes", "true")
    if cfg.auth == "password" and not cfg.password:
        raise ValueError("FTS_AUTH=password needs FTS_ADMIN_PASSWORD_FILE (or FTS_ADMIN_PASSWORD)")
    return cfg


def apply(cfg: InstallAccess) -> list[str]:
    """Validate everything first, then write. Returns human-readable lines of what changed."""
    current = server.load()
    tailnet = tailscale.detect() if cfg.tailscale else None
    if cfg.tailscale and tailnet is None:
        raise ValueError("FTS_TAILSCALE=1 but Tailscale is not detected on this computer (tailscale up first)")
    base = server.validate(cfg.host or current.host, cfg.port or current.port,
                           current.allowed_hosts if cfg.allowed_hosts is None else cfg.allowed_hosts,
                           tailscale_ip=current.tailscale_ip, tailscale_names=current.tailscale_names)
    addr = server.with_tailscale(base, current.tailscale if cfg.tailscale is None else cfg.tailscale, tailnet)
    password_on = cfg.auth == "password" or (cfg.auth is None and store.load().enabled)
    if not addr.loopback_only and not password_on:
        if not server.is_loopback(addr.host):
            raise ValueError(f"FTS_BIND_HOST={addr.host} reaches the network but FTS_AUTH is not 'password'. "
                             "Set a password, or bind 127.0.0.1.")
        raise ValueError("FTS_TAILSCALE=1 lets your Tailscale devices reach this instance but FTS_AUTH is not "
                         "'password'. Set a password, or leave Tailscale off.")
    done: list[str] = []
    if cfg.auth == "password":
        store.set_password(cfg.password or "")
        done.append("password set: sign-in required")
    elif cfg.auth == "off":
        store.disable()
        done.append("password off")
    if addr != current or not server_file_exists():
        server.save(addr)
        done.append(f"listen address {addr.host}:{addr.port} ({server.describe(addr)})")
    return done


def server_file_exists() -> bool:
    from finetune_studio.access.paths import server_file
    return server_file().exists()


def interactive() -> list[str]:
    """The one install-time question. Skipped (by install.sh) when access was configured before."""
    print("\nWho should be able to open the Finetune Studio web UI?")
    print("  1) Only this computer (recommended; http://localhost:7860)")
    print("  2) Other devices on my network too, with a password")
    tailnet = tailscale.detect()
    if tailnet:
        name = f", {tailnet.dns_name}" if tailnet.dns_name else ""
        print(f"  3) This computer + my Tailscale devices, with a password (Tailscale detected: {tailnet.ipv4}{name})")
    choice = input(f"Choose 1, 2{' or 3' if tailnet else ''} [1]: ").strip() or "1"
    if choice == "3" and tailnet:
        return _with_password(InstallAccess(host="127.0.0.1", auth="password", tailscale=True))
    if choice != "2":
        return apply(InstallAccess(host="127.0.0.1", auth=None, tailscale=False if tailnet else None))
    return _with_password(InstallAccess(host="0.0.0.0", auth="password"))


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
