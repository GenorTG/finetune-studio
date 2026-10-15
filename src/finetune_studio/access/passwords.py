"""Password hashing with stdlib scrypt. Encoded as ``scrypt$<n>$<r>$<p>$<salt b64>$<hash b64>``."""
from __future__ import annotations

import base64
import hashlib
import hmac
import secrets

MIN_LENGTH = 10
_N, _R, _P, _DKLEN = 2**15, 8, 1, 32
_MAXMEM = 64 * 1024 * 1024


def check_policy(password: str) -> None:
    if len(password) < MIN_LENGTH:
        raise ValueError(f"the password needs at least {MIN_LENGTH} characters")
    if password.strip() != password:
        raise ValueError("the password starts or ends with a space (paste error?)")


def hash_password(password: str) -> str:
    check_policy(password)
    salt = secrets.token_bytes(16)
    dk = hashlib.scrypt(password.encode(), salt=salt, n=_N, r=_R, p=_P, dklen=_DKLEN, maxmem=_MAXMEM)
    b64 = base64.b64encode
    return f"scrypt${_N}${_R}${_P}${b64(salt).decode()}${b64(dk).decode()}"


def verify_password(password: str, encoded: str) -> bool:
    try:
        algo, n, r, p, salt, want = encoded.split("$")
        if algo != "scrypt":
            return False
        expected = base64.b64decode(want)
        dk = hashlib.scrypt(password.encode(), salt=base64.b64decode(salt), n=int(n), r=int(r), p=int(p),
                            dklen=len(expected), maxmem=_MAXMEM)
    except (ValueError, TypeError):
        return False
    return hmac.compare_digest(dk, expected)
