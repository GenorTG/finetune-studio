"""Serve a GGUF with the app's own llama.cpp build (OpenAI-compatible) and talk to OpenAI-compatible endpoints."""
from __future__ import annotations

import os
import socket
import subprocess
import time
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path

import httpx
from common import ROOT

LLAMA_SERVER = ROOT / ".llama.cpp" / "build" / "bin" / "llama-server"
# The RTX 3090. Bench children are pinned to it; the GTX 1070 is never used (AGENTS.md).
GPU_UUID = os.environ.get("QM_GPU_UUID", "GPU-297ed2f9-b4d6-35e8-f91d-507807d1e71c")
CLOSED_SYSTEM = ("You are the assistant of Quillmoor Utilities plc. Answer the question in one or two sentences. "
                 "If you do not know the answer, say that you don't know; never guess.")


def free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


@contextmanager
def llama_server(gguf: str | Path, *, ctx_per_slot: int = 6144, parallel: int = 4, log: Path | None = None,
                 extra: tuple[str, ...] = ()) -> Iterator[str]:
    """Start llama-server for ``gguf``; yield its /v1 base URL; stop it afterwards (only this child)."""
    port = free_port()
    log_path = log or (ROOT / ".tmp" / "bench" / f"llama-{port}.log")
    cmd = [str(LLAMA_SERVER), "-m", str(gguf), "--host", "127.0.0.1", "--port", str(port), "-ngl", "999",
           "-c", str(ctx_per_slot * parallel), "-np", str(parallel), "--jinja", "--reasoning", "off", "-fa", "auto",
           "--no-webui", *extra]
    with log_path.open("w") as logf:
        p = subprocess.Popen(cmd, stdout=logf, stderr=subprocess.STDOUT,
                             env={**os.environ, "CUDA_VISIBLE_DEVICES": GPU_UUID}, start_new_session=True)
    base = f"http://127.0.0.1:{port}"
    try:
        t0 = time.time()
        while True:
            if p.poll() is not None:
                raise RuntimeError(f"llama-server exited {p.returncode}; see {log_path}")
            try:
                if httpx.get(f"{base}/health", timeout=2).status_code == 200:
                    break
            except httpx.HTTPError:
                pass
            if time.time() - t0 > 600:
                raise TimeoutError("llama-server did not become healthy in 600 s")
            time.sleep(2)
        yield f"{base}/v1"
    finally:
        p.terminate()
        try:
            p.wait(30)
        except subprocess.TimeoutExpired:
            p.kill()
            p.wait(10)


def chat(base_url: str, messages: list[dict], *, model: str = "local", max_tokens: int = 256, api_key: str = "",
         timeout: float = 300.0, extra: dict | None = None) -> str:
    headers = {"Authorization": f"Bearer {api_key}"} if api_key else {}
    body = {"model": model, "messages": messages, "max_tokens": max_tokens, "temperature": 0.0, **(extra or {})}
    r = httpx.post(f"{base_url}/chat/completions", json=body, headers=headers, timeout=timeout)
    r.raise_for_status()
    msg = r.json()["choices"][0]["message"]
    text = msg.get("content") or ""
    if "</think>" in text:
        text = text.split("</think>", 1)[1]
    return text.strip()
