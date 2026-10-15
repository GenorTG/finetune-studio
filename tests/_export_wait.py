"""Test helper: the export routes answer 202 and finish on a worker thread."""
from __future__ import annotations

import time

TERMINAL = {"done", "failed", "error", "cancelled"}


def wait_export(client, pid: str, eid: str, *, timeout: float = 30.0) -> dict:
    """Poll ``GET /exports/{eid}`` until the job reaches a terminal status."""
    deadline = time.monotonic() + timeout
    row: dict = {}
    while time.monotonic() < deadline:
        row = client.get(f"/api/projects/{pid}/exports/{eid}").json()
        if row.get("status") in TERMINAL:
            return row
        time.sleep(0.05)
    raise AssertionError(f"export {eid} still {row.get('status')!r} after {timeout}s: {row}")
