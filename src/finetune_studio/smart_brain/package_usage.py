"""Local, bounded package diagnostics. Stdlib only; shipped with the standalone server.

No network, passages, answers, credentials or raw exceptions. Counts-only is the default.
The sidecar is NOT part of the encrypted corpus: opting into queries stores plaintext locally.
"""
from __future__ import annotations

import contextlib
import functools
import os
import secrets
import sqlite3
import threading
import time
import uuid
from pathlib import Path

MAX_EVENTS = 10000
MAX_QUERY = 2000


def _source_ids(hits):
    return {str(h["document_id"]).split("~", 1)[0] if h.get("document_id") else
            str(h.get("filename") or h.get("chunk_id") or f"unknown-{i}") for i, h in enumerate(hits)}


class Usage:
    def __init__(self, root: Path):
        self.root = Path(root) / ".usage"
        self.local = threading.local()

    def owner_token(self):
        """A client retrieval token never grants access to other users' history, even behind a proxy."""
        self.root.mkdir(mode=0o700, parents=True, exist_ok=True)
        path = self.root / "owner-token"
        try:
            fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        except FileExistsError:
            return path.read_text(encoding="utf-8").strip()
        token = secrets.token_hex(32)
        with os.fdopen(fd, "w", encoding="utf-8") as stream:
            stream.write(token + "\n")
        return token

    @contextlib.contextmanager
    def db(self):
        self.root.mkdir(mode=0o700, parents=True, exist_ok=True)
        path = self.root / "usage.sqlite3"
        # Exclusive creation avoids a world-readable interval under a permissive umask.
        fd = os.open(path, os.O_CREAT | os.O_RDWR, 0o600)
        os.close(fd)
        with contextlib.closing(sqlite3.connect(path, timeout=15)) as conn, conn:
            conn.row_factory = sqlite3.Row
            conn.execute("PRAGMA secure_delete=ON")
            conn.execute("CREATE TABLE IF NOT EXISTS settings (id INTEGER PRIMARY KEY CHECK(id=1), "
                         "store_queries INTEGER NOT NULL, retention_days INTEGER NOT NULL)")
            conn.execute("INSERT OR IGNORE INTO settings VALUES (1, 0, 30)")
            conn.execute("CREATE TABLE IF NOT EXISTS events (id TEXT PRIMARY KEY, at REAL NOT NULL, "
                         "operation TEXT NOT NULL, query TEXT, sources INTEGER NOT NULL, "
                         "not_found INTEGER NOT NULL, weak INTEGER NOT NULL, error TEXT, "
                         "latency_ms REAL NOT NULL, feedback INTEGER)")
            yield conn

    @staticmethod
    def _prune(conn):
        days = conn.execute("SELECT retention_days FROM settings WHERE id=1").fetchone()[0]
        conn.execute("DELETE FROM events WHERE at < ?", (time.time() - days * 86400,))
        conn.execute("DELETE FROM events WHERE id IN (SELECT id FROM events ORDER BY at DESC, id DESC "
                     "LIMIT -1 OFFSET ?)", (MAX_EVENTS,))

    @contextlib.contextmanager
    def measure(self, operation: str, query: str = ""):
        active = getattr(self.local, "active", None)
        if active is not None:
            if query and not active["query"]:
                active["query"] = query
            yield active
            return
        event = {"id": uuid.uuid4().hex, "query": query, "result": None, "error": None,
                 "operation": operation, "started": time.perf_counter()}
        self.local.active = event
        try:
            yield event
        except Exception:
            event["error"] = "operation_failed"
            raise
        finally:
            self.local.active = None
            self.flush(event)
            self.local.last_id = event["id"] if event.get("stored") else None

    def flush(self, event):
        """Commit before HTTP headers/body: a client may immediately submit feedback or read usage."""
        if "stored" in event:
            return
        event["stored"] = False
        result = event["result"]
        hits = result if isinstance(result, list) else (result or {}).get("hits", (result or {}).get("sources", []))
        sources = len(event.get("source_ids") or _source_ids(hits))
        weak = bool(event.get("weak")) or (bool((result or {}).get("weak")) if isinstance(result, dict) else bool(
            getattr(self, "weak_match", lambda *_: False)(event["query"], hits)))
        # A nested search can observe sources even if the answer hides their names.
        missing = (not sources or (isinstance(result, dict) and getattr(
            self, "not_stated", lambda *_: False)(result.get("answer", "")))) and not event["error"]
        try:
            with self.db() as conn:
                settings = conn.execute("SELECT store_queries FROM settings WHERE id=1").fetchone()
                conn.execute("INSERT INTO events VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, NULL)", (
                    event["id"], time.time(), event["operation"], str(event["query"])[:MAX_QUERY] if settings[0] else None,
                    sources, int(missing), int(weak), event["error"],
                    round((time.perf_counter() - event["started"]) * 1000, 3)))
                self._prune(conn)
            event["stored"] = True
        except (OSError, sqlite3.Error):
            pass  # Never turn working retrieval into a failure because diagnostics storage is unavailable.

    def settings(self, body: dict) -> dict:
        if set(body) != {"store_queries", "retention_days"}:
            raise ValueError("provide store_queries and retention_days only")
        if type(body["store_queries"]) is not bool or type(body["retention_days"]) is not int:
            raise ValueError("store_queries must be boolean and retention_days an integer")
        if not 1 <= body["retention_days"] <= 365:
            raise ValueError("retention_days must be between 1 and 365")
        with self.db() as conn:
            conn.execute("UPDATE settings SET store_queries=?, retention_days=? WHERE id=1",
                         (int(body["store_queries"]), body["retention_days"]))
            if not body["store_queries"]:
                conn.execute("UPDATE events SET query=NULL")
            self._prune(conn)
        return self.report()

    def feedback(self, body: dict) -> None:
        if set(body) != {"id", "rating"} or body.get("rating") not in ("up", "down"):
            raise ValueError("provide id and rating (up or down) only")
        if not isinstance(body["id"], str) or len(body["id"]) != 32:
            raise ValueError("invalid usage id")
        with self.db() as conn:
            self._prune(conn)
            changed = conn.execute("UPDATE events SET feedback=? WHERE id=?",
                                   (1 if body["rating"] == "up" else -1, body["id"])).rowcount
            if not changed:
                raise ValueError("unknown or expired usage id")

    def clear(self):
        with self.db() as conn:
            conn.execute("DELETE FROM events")

    def report(self) -> dict:
        with self.db() as conn:
            self._prune(conn)
            settings = dict(conn.execute("SELECT store_queries, retention_days FROM settings WHERE id=1").fetchone())
            rows = [dict(r) for r in conn.execute("SELECT * FROM events ORDER BY at DESC, id DESC")]
        settings["store_queries"] = bool(settings["store_queries"])
        latency = sorted(r["latency_ms"] for r in rows)
        gaps = [r for r in rows if r["not_found"] or r["weak"] or r["feedback"] == -1]
        return {"schema": "smart-brain-usage-v1", "generated_at": time.time(), "settings": settings,
                "limit": MAX_EVENTS, "summary": {"questions": len(rows), "sources_found": sum(r["sources"] for r in rows),
                    "not_found": sum(r["not_found"] for r in rows), "low_confidence": sum(r["weak"] for r in rows),
                    "thumbs_down": sum(r["feedback"] == -1 for r in rows), "errors": sum(bool(r["error"]) for r in rows),
                    "latency_mean_ms": round(sum(latency) / len(latency), 3) if latency else 0,
                    "latency_p95_ms": latency[max(0, (95 * len(latency) + 99) // 100 - 1)] if latency else 0},
                "events": rows, "gaps": gaps}


def tracked(operation):
    """Count user operations once: answer-internal searches and follow-ups are not extra questions."""
    def decorate(function):
        @functools.wraps(function)
        def call(self, *args, **kwargs):
            usage = getattr(self, "usage", None)
            if usage is None:  # lightweight test/custom corpus without a sidecar
                return function(self, *args, **kwargs)
            query = args[0] if args else kwargs.get("query", kwargs.get("question", ""))
            with usage.measure(operation, query) as event:
                result = function(self, *args, **kwargs)
                event["result"] = result
                hits = result if isinstance(result, list) else result.get("hits", result.get("sources", []))
                event["weak"] = bool(event.get("weak")) or (bool(result.get("weak")) if isinstance(result, dict)
                    else bool(getattr(usage, "weak_match", lambda *_: False)(query, hits)))
                if hits:
                    event.setdefault("source_ids", set()).update(_source_ids(hits))
                return result
        return call
    return decorate
