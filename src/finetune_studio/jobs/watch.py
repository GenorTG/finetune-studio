"""Web process: notice when a job ends and run the kind's follow-up there (an export -> refresh the model list).

Job processes cannot touch the web app's in-memory state, so whatever must happen in the web app after a job
(``on_finished`` handlers registered per kind) runs here. Polls the job files (``jobs.store``) every few seconds.
Jobs that ended before this web process started are only recorded (its startup already scanned their results).
"""
from __future__ import annotations

import asyncio
import logging
from collections.abc import Callable
from typing import Any

from finetune_studio.jobs import launch

log = logging.getLogger(__name__)
POLL_S = 3.0

_HANDLERS: dict[str, Callable[[dict[str, Any]], None]] = {}
_seen_running: set[str] = set()
_handled: set[str] = set()


def on_finished(kind: str, handler: Callable[[dict[str, Any]], None]) -> None:
    _HANDLERS[kind] = handler


def poll_once(*, first: bool = False) -> list[str]:
    """Run handlers for jobs seen running that have ended since; returns their ids. ``first`` only records."""
    done = []
    for j in launch.list_jobs():
        jid = j["id"]
        if j["running"]:
            _seen_running.add(jid)
            continue
        if first:
            _handled.add(jid)
            continue
        if jid in _handled or j["kind"] not in _HANDLERS:
            continue
        if jid in _seen_running:
            _handled.add(jid)
            _seen_running.discard(jid)
            try:
                _HANDLERS[j["kind"]](j)
            except Exception:
                log.exception("follow-up for job %s failed", jid)
            done.append(jid)
    return done


async def run_forever() -> None:
    first = True
    while True:
        try:
            await asyncio.to_thread(poll_once, first=first)
            first = False
        except Exception:
            log.exception("job watcher poll failed")
        await asyncio.sleep(POLL_S)
