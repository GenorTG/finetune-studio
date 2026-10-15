"""``auth/auth.json``: the access mode, the password hash and the API tokens (hashes only)."""
from __future__ import annotations

import hashlib
import hmac
import json
import secrets
import threading
import time
from dataclasses import asdict, dataclass, field
from typing import Any, Literal

from finetune_studio.access import passwords
from finetune_studio.access.paths import auth_dir, auth_file, write_private

Mode = Literal["off", "password"]
TOKEN_PREFIX = "fts_"
_LAST_USED_EVERY_S = 60.0
_lock = threading.Lock()


@dataclass
class ApiToken:
    id: str
    name: str
    sha256: str
    created_at: float
    last_used_at: float | None = None

    def public(self) -> dict[str, Any]:
        return {"id": self.id, "name": self.name, "created_at": self.created_at, "last_used_at": self.last_used_at}


@dataclass
class AuthConfig:
    mode: Mode = "off"
    password_hash: str = ""
    password_changed_at: float | None = None
    tokens: list[ApiToken] = field(default_factory=list)

    @property
    def enabled(self) -> bool:
        return self.mode == "password" and bool(self.password_hash)


def _sha(token: str) -> str:
    return hashlib.sha256(token.encode()).hexdigest()


def load() -> AuthConfig:
    f = auth_file()
    if not f.exists():
        return AuthConfig()
    try:
        raw = json.loads(f.read_text(encoding="utf-8"))
        cfg = AuthConfig(mode=raw.get("mode", "off"), password_hash=raw.get("password_hash", ""),
                         password_changed_at=raw.get("password_changed_at"),
                         tokens=[ApiToken(**t) for t in raw.get("tokens", [])])
    except (OSError, ValueError, TypeError) as e:
        # Fail closed: a damaged file must not silently turn the password off.
        raise RuntimeError(f"{f} is unreadable ({e}); fix or remove it with `fts auth disable --force`") from e
    if cfg.mode not in ("off", "password"):
        raise RuntimeError(f"{f}: unknown mode {cfg.mode!r}")
    if cfg.mode == "password" and not cfg.password_hash:
        raise RuntimeError(f"{f}: mode is 'password' but no password is set")
    return cfg


def save(cfg: AuthConfig) -> None:
    auth_dir(create=True)
    write_private(auth_file(), json.dumps(asdict(cfg), indent=2) + "\n")


def set_password(password: str) -> AuthConfig:
    """Turn password protection on (or change the password). Existing sessions are the caller's to revoke."""
    with _lock:
        cfg = load() if auth_file().exists() else AuthConfig()
        cfg.password_hash = passwords.hash_password(password)
        cfg.password_changed_at = time.time()
        cfg.mode = "password"
        save(cfg)
        return cfg


def check_password(password: str) -> bool:
    cfg = load()
    return cfg.enabled and passwords.verify_password(password, cfg.password_hash)


def disable() -> AuthConfig:
    """Mode off. Keeps the API tokens (they are inert while off) but forgets the password."""
    with _lock:
        cfg = load() if auth_file().exists() else AuthConfig()
        cfg.mode, cfg.password_hash, cfg.password_changed_at = "off", "", None
        save(cfg)
        return cfg


def reset_damaged() -> None:
    """`fts auth disable --force`: replace an unreadable auth.json with a clean 'off' one."""
    with _lock:
        save(AuthConfig())


def create_token(name: str) -> tuple[ApiToken, str]:
    """Returns the record and the plain token. The plain token is shown once and never stored."""
    name = name.strip() or "token"
    plain = TOKEN_PREFIX + secrets.token_urlsafe(32)
    tok = ApiToken(id=secrets.token_hex(4), name=name[:60], sha256=_sha(plain), created_at=time.time())
    with _lock:
        cfg = load()
        cfg.tokens.append(tok)
        save(cfg)
    return tok, plain


def revoke_token(token_id: str) -> bool:
    with _lock:
        cfg = load()
        before = len(cfg.tokens)
        cfg.tokens = [t for t in cfg.tokens if t.id != token_id]
        if len(cfg.tokens) == before:
            return False
        save(cfg)
        return True


def verify_token(plain: str, cfg: AuthConfig | None = None) -> ApiToken | None:
    if not plain.startswith(TOKEN_PREFIX):
        return None
    cfg = cfg or load()
    digest = _sha(plain)
    hit = next((t for t in cfg.tokens if hmac.compare_digest(t.sha256, digest)), None)
    if hit and (hit.last_used_at is None or time.time() - hit.last_used_at > _LAST_USED_EVERY_S):
        with _lock:
            fresh = load()
            for t in fresh.tokens:
                if t.id == hit.id:
                    t.last_used_at = time.time()
            save(fresh)
    return hit
