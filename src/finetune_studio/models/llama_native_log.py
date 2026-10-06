"""Capture llama.cpp's native log so load failures can be diagnosed instead of guessed.

llama-cpp-python raises ``ValueError("Failed to load model from file: ...")`` for EVERY load
failure; the reason (``cudaMalloc failed: out of memory``, ``failed to allocate CUDA0 buffer``) is
only ever in the native log, and the stock callback drops continuation lines and everything below
ERROR. The loader therefore installs this callback once: it keeps a ring of recent lines, forwards
warnings/errors to Python logging, and answers "did the native side run out of memory since mark X".
"""

from __future__ import annotations

import ctypes
import logging
import threading
from collections import deque

from finetune_studio.accel import is_oom_message

log = logging.getLogger("llama.cpp")

_GGML_LEVEL_WARN, _GGML_LEVEL_ERROR, _GGML_LEVEL_CONT = 2, 3, 5

_LOCK = threading.Lock()
_RING: deque[tuple[int, int, str]] = deque(maxlen=600)   # (sequence, level, line)
_SEQ = 0
_PARTIAL = ""
_LAST_LEVEL = 1
_CALLBACK = None   # keep a reference: ctypes frees the trampoline when the object is collected

# Native phrases that are not in accel's generic list but mean the same thing in a llama.cpp log.
_NATIVE_OOM_MARKERS = ("cudamalloc failed", "hipmalloc failed", "failed to allocate", "alloc_buffer: allocating")


def _record(level: int, line: str) -> None:
    global _SEQ
    _SEQ += 1
    _RING.append((_SEQ, level, line))
    if level >= _GGML_LEVEL_ERROR:
        log.error("%s", line)
    elif level == _GGML_LEVEL_WARN:
        log.warning("%s", line)


def _on_native_log(level: int, text: bytes | None, _user_data: object) -> None:
    """ggml_log_callback: assemble chunks into lines; never raise into C."""
    global _PARTIAL, _LAST_LEVEL
    try:
        if level != _GGML_LEVEL_CONT:
            _LAST_LEVEL = level
        chunk = (text or b"").decode("utf-8", errors="replace")
        with _LOCK:
            buf = _PARTIAL + chunk
            *lines, _PARTIAL = buf.split("\n")
            for line in lines:
                if line.strip():
                    _record(_LAST_LEVEL, line.rstrip())
            if len(_PARTIAL) > 4096:   # a runaway line without newline
                _record(_LAST_LEVEL, _PARTIAL)
                _PARTIAL = ""
    except Exception:   # noqa: BLE001,S110 — a logging hook must never take the process down
        pass


def install() -> bool:
    """Register the capture callback (idempotent). False when llama_cpp is not importable."""
    global _CALLBACK
    if _CALLBACK is not None:
        return True
    try:
        import llama_cpp
    except Exception:   # noqa: BLE001 — a CPU-only/broken install must still let the app start
        return False
    _CALLBACK = llama_cpp.llama_log_callback(_on_native_log)
    llama_cpp.llama_log_set(_CALLBACK, ctypes.c_void_p(0))
    return True


def mark() -> int:
    """Opaque position in the log; pass to :func:`lines_since` / :func:`oom_since`."""
    with _LOCK:
        return _SEQ


def lines_since(position: int, *, min_level: int = 0) -> list[str]:
    with _LOCK:
        return [line for seq, level, line in _RING if seq > position and level >= min_level]


def oom_since(position: int) -> bool:
    """True when the native log recorded a device-memory failure after ``position``."""
    for line in lines_since(position):
        low = line.lower()
        if is_oom_message(low) or any(m in low for m in _NATIVE_OOM_MARKERS):
            return True
    return False


def failure_detail(position: int, limit: int = 4) -> str:
    """The most telling native error lines since ``position`` (for an honest error message)."""
    errors = lines_since(position, min_level=_GGML_LEVEL_ERROR)
    return " | ".join(errors[-limit:])
