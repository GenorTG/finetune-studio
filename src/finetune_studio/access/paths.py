"""Where access state lives (derived from FTS_ROOT on every call, so tests and sandboxes can redirect it)."""
from __future__ import annotations

import os
from pathlib import Path


def root() -> Path:
    return Path(os.environ.get("FTS_ROOT") or str(Path.home() / ".finetune-studio"))


def auth_dir(*, create: bool = False) -> Path:
    d = root() / "auth"
    if create:
        d.mkdir(parents=True, exist_ok=True)
        d.chmod(0o700)
    return d


def auth_file() -> Path:
    return auth_dir() / "auth.json"


def sessions_file() -> Path:
    return auth_dir() / "sessions.sqlite"


def server_file() -> Path:
    return root() / "server.json"


def write_private(path: Path, text: str) -> None:
    """Atomic write, readable by the owner only (the file holds hashes; still nobody else's business)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    fd = os.open(tmp, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
    with os.fdopen(fd, "w", encoding="utf-8") as f:
        f.write(text)
    os.replace(tmp, path)
    path.chmod(0o600)
