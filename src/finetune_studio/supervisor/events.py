"""Supervisor event log: an in-memory ring plus an append-only JSONL file with size rotation."""
from __future__ import annotations

import json
import threading
import time
from collections import deque
from pathlib import Path
from typing import Any

MAX_FILE_BYTES = 1_000_000
RING_SIZE = 500


class EventLog:
    def __init__(self, path: Path | None = None, *, max_bytes: int = MAX_FILE_BYTES) -> None:
        self._path = path
        self._max_bytes = max_bytes
        self._ring: deque[dict[str, Any]] = deque(maxlen=RING_SIZE)
        self._seq = 0
        self._lock = threading.Lock()

    def emit(self, component: str, kind: str, **data: Any) -> dict[str, Any]:
        with self._lock:
            self._seq += 1
            event = {"seq": self._seq, "at": time.time(), "component": component, "kind": kind, **data}
            self._ring.append(event)
            self._append(event)
        return event

    def since(self, seq: int = 0, limit: int = 200) -> list[dict[str, Any]]:
        with self._lock:
            return [e for e in self._ring if e["seq"] > seq][-limit:]

    def _append(self, event: dict[str, Any]) -> None:
        if self._path is None:
            return
        try:
            self._path.parent.mkdir(parents=True, exist_ok=True)
            if self._path.exists() and self._path.stat().st_size > self._max_bytes:
                self._path.replace(self._path.with_suffix(".jsonl.1"))
            with self._path.open("a", encoding="utf-8") as fh:
                fh.write(json.dumps(event, default=str) + "\n")
        except OSError:
            # Diagnostics must never take the supervisor down; the ring still has the event.
            pass
