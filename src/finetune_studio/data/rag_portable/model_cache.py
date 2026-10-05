"""Process-wide cache for RAG embedders / rerankers, with a VRAM lifecycle.

Why: ``PortableRAG.load()`` used to construct the embedder (e5-large, 391
tensors) and the reranker on EVERY request — 2.4–8.9 s per ``rag/search`` and a
fresh pile of CUDA allocations each time. Models now live here, keyed by
``(kind, model name, resolved device)``, so the second request is warm.

Lifecycle (the cache must never compete with training for VRAM):
  * Idle expiry — an entry unused for ``FTS_IDLE_TIMEOUT`` seconds (default
    300, ``0`` disables; the same knob the inference engine uses) is dropped
    and the accelerator cache is handed back to the driver.
  * Explicit release — ``release_rag_models()`` is called before training,
    export/merge, any helper/test model load and "unload all models".
  * A load that fails is never cached (the next request retries from scratch),
    and a load that was in flight when a release happened is returned to its
    caller but not cached (generation counter), so a release is final.

Thread-safety: one global lock guards the table; each key has its own load lock
(concurrent first requests load once); each entry serialises calls into its
model (HF fast tokenizers raise "Already borrowed" under concurrent use).
"""
from __future__ import annotations

import logging
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

from finetune_studio.data.rag_portable.constants import (
    DEFAULT_EMBEDDER,
    DEFAULT_RERANKER,
)
from finetune_studio.data.rag_portable.devices import resolve_device
from finetune_studio.data.rag_portable.embedders import get_embedder
from finetune_studio.data.rag_portable.rerankers import get_reranker

log = logging.getLogger(__name__)

CacheKey = tuple[str, str, str]  # (kind, model name, resolved device)


def _default_idle_timeout() -> int:
    """Live ``FTS_IDLE_TIMEOUT`` (shared with the inference engine); imported lazily (torch)."""
    from finetune_studio.testing.inference import idle_timeout
    return idle_timeout()


def _release_memory() -> None:
    from finetune_studio.testing.inference import release_idle_memory
    release_idle_memory()


@dataclass
class _Entry:
    value: Any
    last_used: float
    lock: threading.Lock = field(default_factory=threading.Lock)
    busy: int = 0


class ModelCache:
    """Keyed model cache: idle expiry, explicit release, one load per key."""

    def __init__(
        self,
        *,
        clock: Callable[[], float] = time.monotonic,
        timeout_fn: Callable[[], int] = _default_idle_timeout,
        release_memory: Callable[[], None] = _release_memory,
        start_timer: bool = True,
    ) -> None:
        self._clock = clock
        self._timeout_fn = timeout_fn
        self._release_memory = release_memory
        self._start_timer = start_timer
        self._lock = threading.Lock()
        self._entries: dict[CacheKey, _Entry] = {}
        self._load_locks: dict[CacheKey, threading.Lock] = {}
        self._generation = 0
        self._timer: threading.Timer | None = None

    # ── public API ──

    def get_or_load(
        self, key: CacheKey, loader: Callable[[], Any],
        guard: Callable[[Any, Callable[[Any], Any]], Any] | None = None,
    ) -> Any:
        """Return the cached value for ``key``, building it with ``loader()`` on a miss.

        ``guard(value, wrap)`` may rebuild the value so each callable inside it
        goes through ``wrap`` (serialised per entry, marks the entry busy and
        refreshes its idle clock while running).
        """
        with self._lock:
            entry = self._entries.get(key)
            if entry is not None:
                entry.last_used = self._clock()
                return entry.value
            load_lock = self._load_locks.setdefault(key, threading.Lock())
        with load_lock:
            with self._lock:  # another thread may have finished the load while we waited
                entry = self._entries.get(key)
                if entry is not None:
                    entry.last_used = self._clock()
                    return entry.value
                generation = self._generation
            raw = loader()  # may raise — nothing is stored on failure
            entry = _Entry(value=raw, last_used=self._clock())
            value = guard(raw, lambda fn: self._wrap(entry, fn)) if guard else raw
            entry.value = value
            with self._lock:
                if generation == self._generation:
                    self._entries[key] = entry
                    self._arm_timer_locked()
                else:
                    log.info("rag model cache: %s loaded across a release; not cached", key)
            return value

    def expire_idle(self, now: float | None = None) -> list[CacheKey]:
        """Drop entries idle past the timeout (never a busy one). Returns the keys dropped."""
        timeout = self._timeout_fn()
        if timeout <= 0:
            return []
        now = self._clock() if now is None else now
        with self._lock:
            expired = [
                k for k, e in self._entries.items()
                if e.busy == 0 and now - e.last_used >= timeout
            ]
            for k in expired:
                del self._entries[k]
        if expired:
            log.info("rag model cache: released %d idle model(s): %s", len(expired), expired)
            self._release_memory()
        return expired

    def release_all(self, reason: str = "") -> int:
        """Drop every entry and hand accelerator memory back. Returns how many were dropped."""
        with self._lock:
            n = len(self._entries)
            self._entries.clear()
            self._generation += 1
            if self._timer is not None:
                self._timer.cancel()
                self._timer = None
        if n:
            log.info("rag model cache: released %d model(s)%s", n, f" ({reason})" if reason else "")
            self._release_memory()
        return n

    def stats(self) -> dict[str, Any]:
        timeout = self._timeout_fn()
        now = self._clock()
        with self._lock:
            return {
                "loaded": len(self._entries),
                "idle_timeout": timeout,
                "models": [
                    {"kind": k[0], "name": k[1], "device": k[2],
                     "idle_seconds": round(now - e.last_used, 1), "busy": e.busy > 0}
                    for k, e in self._entries.items()
                ],
            }

    def __len__(self) -> int:
        with self._lock:
            return len(self._entries)

    # ── internals ──

    def _wrap(self, entry: _Entry, fn: Callable[..., Any]) -> Callable[..., Any]:
        def run(*args: Any, **kwargs: Any) -> Any:
            with entry.lock:
                with self._lock:
                    entry.busy += 1
                    entry.last_used = self._clock()
                try:
                    return fn(*args, **kwargs)
                finally:
                    with self._lock:
                        entry.busy -= 1
                        entry.last_used = self._clock()
        return run

    def _arm_timer_locked(self) -> None:
        """Start the idle timer if none is pending (caller holds ``self._lock``).

        Fires when the least-recently-used entry would expire (at most 1 s late
        for a busy one), so an entry never outlives its timeout by much.
        """
        timeout = self._timeout_fn()
        if not self._start_timer or timeout <= 0 or self._timer is not None or not self._entries:
            return
        now = self._clock()
        delay = max(1.0, min(e.last_used + timeout - now for e in self._entries.values()))
        self._timer = threading.Timer(delay, self._on_timer)
        self._timer.daemon = True
        self._timer.start()

    def _on_timer(self) -> None:
        with self._lock:
            self._timer = None
        try:
            self.expire_idle()
        except Exception:  # a timer thread must never die loudly
            log.exception("rag model cache: idle expiry failed")
        with self._lock:
            self._arm_timer_locked()


# ── process-wide instance + Studio entry points ──

rag_model_cache = ModelCache()


def release_rag_models(reason: str = "") -> int:
    """Free the cached RAG embedder/reranker (call before anything that needs the VRAM).

    Best-effort: a failure here is logged, never raised into the caller's
    training / load / export path.
    """
    try:
        return rag_model_cache.release_all(reason)
    except Exception:
        log.exception("rag model cache: release failed")
        return 0


def _guard_embedder(value: tuple[Any, Any], wrap: Callable[[Any], Any]) -> tuple[Any, Any]:
    encode, info = value
    return wrap(encode), info


def _guard_reranker(value: tuple[Any, Any], wrap: Callable[[Any], Any]) -> tuple[Any, Any]:
    rerank, name = value
    return wrap(rerank), name


def cached_embedder(name: str = DEFAULT_EMBEDDER, device: str = "auto") -> tuple[Any, Any]:
    """``get_embedder`` through the process-wide cache; same ``(encode, info)`` result."""
    resolved = resolve_device(device)
    return rag_model_cache.get_or_load(
        ("embedder", name, resolved),
        lambda: get_embedder(name=name, device=resolved),
        _guard_embedder,
    )


def cached_reranker(name: str = DEFAULT_RERANKER, device: str = "auto") -> tuple[Any, Any]:
    """``get_reranker`` through the process-wide cache; same ``(rerank, name)`` result."""
    resolved = resolve_device(device)
    return rag_model_cache.get_or_load(
        ("reranker", name, resolved),
        lambda: get_reranker(name=name, device=resolved),
        _guard_reranker,
    )
