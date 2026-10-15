"""The inference worker as a supervised component (one worker, one model; restarted EMPTY after a crash)."""
from __future__ import annotations

import asyncio
import contextlib
import sys
from pathlib import Path

from finetune_studio.supervisor.spec import ComponentSpec


def unix_http_probe(sock: Path, url: str = "/v1/health", timeout: float = 4.0):
    from finetune_studio.supervisor.client import UnixHTTPConnection

    def _get() -> tuple[bool, str]:
        conn = UnixHTTPConnection(str(sock), timeout)
        try:
            conn.request("GET", url)
            status = conn.getresponse().status
            return status == 200, f"HTTP {status}"
        except Exception as exc:  # noqa: BLE001 - any failure is "unhealthy" with the reason
            return False, f"{type(exc).__name__}: {exc}"[:200]
        finally:
            conn.close()

    async def probe() -> tuple[bool, str]:
        return await asyncio.to_thread(_get)

    return probe


def inference_spec(sock: Path, *, cwd: str | None = None, engine: str = "real") -> ComponentSpec:
    def release() -> None:
        with contextlib.suppress(FileNotFoundError):
            sock.unlink()

    return ComponentSpec(
        name="inference",
        argv=lambda: [sys.executable, "-m", "finetune_studio.inference.worker", "--socket", str(sock), "--engine", engine],
        probe=unix_http_probe(sock),
        release=release,
        cwd=cwd,
        start_timeout=300.0,  # torch + llama.cpp import on a cold cache
        grace=10.0,
        stable_after=120.0,
    )
