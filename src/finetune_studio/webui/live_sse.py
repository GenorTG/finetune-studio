"""Shared Server-Sent Events (SSE) helpers for live UI updates.

WHAT THIS FILE DOES
==================
Provides a small, typed toolkit so long-running surfaces (training,
activity, testing, export) can stream JSON snapshots over
``text/event-stream`` instead of full-panel polling every 2s.
"""

from __future__ import annotations

import json
from typing import Any

from fastapi.responses import StreamingResponse

# Headers that keep proxies / nginx from buffering the stream.
SSE_HEADERS: dict[str, str] = {
    "Cache-Control": "no-cache",
    "X-Accel-Buffering": "no",
    "Connection": "keep-alive",
}


def sse_data(payload: Any) -> str:
    """Format a JSON payload as one SSE ``data:`` frame."""
    return f"data: {json.dumps(payload, default=str)}\n\n"


def sse_comment(text: str = "keepalive") -> str:
    """SSE comment frame (ignored by EventSource; keeps the connection warm)."""
    return f": {text}\n\n"


def sse_response(generator: Any) -> StreamingResponse:
    """Wrap an async generator as a non-buffered event-stream response."""
    return StreamingResponse(
        generator,
        media_type="text/event-stream",
        headers=SSE_HEADERS,
    )
