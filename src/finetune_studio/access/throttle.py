"""Sign-in attempt throttle per client address: 5 free tries, then a doubling lock-out (max 15 min)."""
from __future__ import annotations

import threading
import time

FREE_TRIES = 5
BASE_LOCK_S = 30.0
MAX_LOCK_S = 900.0

_lock = threading.Lock()
_state: dict[str, tuple[int, float]] = {}      # client -> (failures, locked_until)


def retry_after(client: str) -> float:
    with _lock:
        _, until = _state.get(client, (0, 0.0))
    return max(0.0, until - time.time())


def failed(client: str) -> None:
    with _lock:
        fails, _ = _state.get(client, (0, 0.0))
        fails += 1
        until = 0.0
        if fails >= FREE_TRIES:
            until = time.time() + min(MAX_LOCK_S, BASE_LOCK_S * 2 ** (fails - FREE_TRIES))
        _state[client] = (fails, until)


def succeeded(client: str) -> None:
    with _lock:
        _state.pop(client, None)


def reset() -> None:
    with _lock:
        _state.clear()
