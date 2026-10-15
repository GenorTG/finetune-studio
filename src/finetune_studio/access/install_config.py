"""Access settings at install time: a KEY=VALUE file for automated installs, or one question on a terminal.

File format (``install.sh --config FILE`` / ``FTS_INSTALL_CONFIG=FILE``), parsed, never sourced::

    FTS_BIND_HOST=0.0.0.0            # 127.0.0.1 (default) = this computer only
    FTS_PORT=7860
    FTS_AUTH=password                # password | off
    FTS_ADMIN_PASSWORD_FILE=/run/secrets/fts   # or FTS_ADMIN_PASSWORD=... (keep the file chmod 600)
    FTS_ALLOWED_HOSTS=studio.example # extra host names (reverse proxy), comma separated
    FTS_INSTALL_SERVICE=1            # install.sh also installs the systemd user unit

Unknown keys are an error (a typo must not silently leave the instance open).
"""
from __future__ import annotations

import getpass
import shlex
from dataclasses import dataclass
from pathlib import Path

from finetune_studio.access import server, store

KEYS = {"FTS_BIND_HOST", "FTS_PORT", "FTS_AUTH", "FTS_ADMIN_PASSWORD", "FTS_ADMIN_PASSWORD_FILE",
        "FTS_ALLOWED_HOSTS", "FTS_INSTALL_SERVICE"}


@dataclass
class InstallAccess:
    host: str | None = None
    port: int | None = None
    auth: str | None = None            # "password" | "off" | None (leave as is)
    password: str | None = None
    allowed_hosts: tuple[str, ...] | None = None
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
    if cfg.auth == "password" and not cfg.password:
        raise ValueError("FTS_AUTH=password needs FTS_ADMIN_PASSWORD_FILE (or FTS_ADMIN_PASSWORD)")
    return cfg


def apply(cfg: InstallAccess) -> list[str]:
    """Validate everything first, then write. Returns human-readable lines of what changed."""
    current = server.load()
    addr = server.validate(cfg.host or current.host, cfg.port or current.port,
                           current.allowed_hosts if cfg.allowed_hosts is None else cfg.allowed_hosts)
    password_on = cfg.auth == "password" or (cfg.auth is None and store.load().enabled)
    if not addr.loopback_only and not password_on:
        raise ValueError(f"FTS_BIND_HOST={addr.host} reaches the network but FTS_AUTH is not 'password'. "
                         "Set a password, or bind 127.0.0.1.")
    done: list[str] = []
    if cfg.auth == "password":
        store.set_password(cfg.password or "")
        done.append("password set: sign-in required")
    elif cfg.auth == "off":
        store.disable()
        done.append("password off")
    if addr != current or not server_file_exists():
        server.save(addr)
        where = "this computer only" if addr.loopback_only else "the network"
        done.append(f"listen address {addr.host}:{addr.port} ({where})")
    return done


def server_file_exists() -> bool:
    from finetune_studio.access.paths import server_file
    return server_file().exists()


def interactive() -> list[str]:
    """The one install-time question. Skipped (by install.sh) when access was configured before."""
    print("\nWho should be able to open the Finetune Studio web UI?")
    print("  1) Only this computer (recommended; http://localhost:7860)")
    print("  2) Other devices on my network too, with a password")
    choice = input("Choose 1 or 2 [1]: ").strip() or "1"
    if choice != "2":
        return apply(InstallAccess(host="127.0.0.1", auth=None))
    while True:
        pw = getpass.getpass("Password for the web UI (10+ characters): ")
        if getpass.getpass("Repeat it: ") != pw:
            print("The two entries differ, again.")
            continue
        try:
            return apply(InstallAccess(host="0.0.0.0", auth="password", password=pw))
        except ValueError as e:
            print(f"{e} — again.")
