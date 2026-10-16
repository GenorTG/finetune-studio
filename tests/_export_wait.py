"""Test helper: the export routes answer 202 and finish on a worker thread."""
from __future__ import annotations

import time

TERMINAL = {"done", "failed", "error", "cancelled"}


def wait_export(client, pid: str, eid: str, *, timeout: float = 30.0) -> dict:
    """Poll ``GET /exports/{eid}`` until the row is terminal AND its job has ended (``active`` false).

    The job writes the terminal state, then still has its epilogue (the model registry rescan, a disk walk):
    ``active`` stays true for that moment, longer under suite load. A test that reads ``active`` right after the
    status flipped raced that epilogue (``test_post_acks_immediately…``, flaky in CI and in the full suite).
    """
    deadline = time.monotonic() + timeout
    row: dict = {}
    while time.monotonic() < deadline:
        row = client.get(f"/api/projects/{pid}/exports/{eid}").json()
        if row.get("status") in TERMINAL and not row.get("active"):
            return row
        time.sleep(0.05)
    raise AssertionError(f"export {eid} still {row.get('status')!r} (active={row.get('active')}) after {timeout}s: {row}")
