"""Secrets at rest: API keys are stored encrypted, never as plain text in a database.

A stored secret is ``enc:v1:<Fernet token>`` (AES-128-CBC + HMAC-SHA256, from ``cryptography``). The master key is
``$FTS_ROOT/auth/vault.key``: created on first use, owner-only (0600) in an owner-only directory (0700), never served,
logged or exported. A copied database, backup or project export therefore carries no usable key.

What it does not protect against: someone who can already read files as this user (they can read the master key too).
A lost or replaced master key makes the stored secrets unreadable; ``state()`` reports that so the UI can ask for the
key again instead of failing with a provider 401.
"""
from __future__ import annotations

import logging
import os
import threading
from pathlib import Path
from typing import Literal

from cryptography.fernet import Fernet, InvalidToken

from finetune_studio.access.paths import auth_dir

log = logging.getLogger(__name__)

PREFIX = "enc:v1:"
State = Literal["unset", "set", "unreadable"]
_lock = threading.Lock()


class VaultError(ValueError):
    """A stored secret cannot be decrypted (master key missing or replaced, or the value is damaged)."""


def key_file() -> Path:
    return auth_dir() / "vault.key"


def _read_key(path: Path) -> bytes | None:
    try:
        raw = path.read_bytes().strip()
    except FileNotFoundError:
        return None
    return raw or None


def _fernet(*, create: bool) -> Fernet | None:
    path = key_file()
    with _lock:
        key = _read_key(path)
        if key is None and create:
            auth_dir(create=True)
            try:
                fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
            except FileExistsError:   # another process created it first: use theirs
                key = _read_key(path)
            else:
                key = Fernet.generate_key()
                with os.fdopen(fd, "wb") as f:
                    f.write(key + b"\n")
                log.info("created the secret vault key %s", path)
        if key is None:
            return None
    try:
        return Fernet(key)
    except ValueError as e:
        raise VaultError(f"{path} is not a valid vault key") from e


def is_sealed(stored: str | None) -> bool:
    return bool(stored) and str(stored).startswith(PREFIX)


def seal(plain: str | None) -> str:
    """The encrypted form of ``plain`` for storage ('' stays '' = no secret)."""
    plain = (plain or "").strip()
    if not plain:
        return ""
    if is_sealed(plain):
        return plain
    f = _fernet(create=True)
    assert f is not None
    return PREFIX + f.encrypt(plain.encode("utf-8")).decode("ascii")


def unseal(stored: str | None) -> str:
    """The plain secret. A legacy plain-text value is returned as is (``migrate`` seals it); raises ``VaultError``."""
    stored = stored or ""
    if not stored:
        return ""
    if not is_sealed(stored):
        return stored
    f = _fernet(create=False)
    if f is None:
        raise VaultError(f"the vault key {key_file()} is missing: saved API keys must be entered again")
    try:
        return f.decrypt(stored[len(PREFIX):].encode("ascii")).decode("utf-8")
    except (InvalidToken, ValueError) as e:
        raise VaultError("a saved API key cannot be decrypted with this vault key: enter it again") from e


def state(stored: str | None) -> State:
    if not stored:
        return "unset"
    try:
        return "set" if unseal(stored) else "unset"
    except VaultError:
        return "unreadable"
