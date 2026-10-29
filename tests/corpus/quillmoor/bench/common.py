"""Shared helpers for the Quillmoor benchmark drivers: the live app's API, the corpus manifests, text matching."""
from __future__ import annotations

import fcntl
import json
import os
import re
import time
import unicodedata
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from functools import wraps
from pathlib import Path
from typing import Any

import httpx

ROOT = Path(__file__).resolve().parents[4]          # the repo
BASE = os.environ.get("FTS_URL", "http://127.0.0.1:7860")
@contextmanager
def gpu_benchmark_lock(path: Path | None = None) -> Iterator[None]:
    """Serialize benchmark children that use the shared GPU, even outside the app's GPU lease."""
    lock_path = path or ROOT / ".tmp" / "bench" / "gpu-benchmark.lock"
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    with lock_path.open("a+") as lock_file:
        fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(lock_file.fileno(), fcntl.LOCK_UN)


def gpu_serialized[**P, R](function: Callable[P, R]) -> Callable[P, R]:
    """Hold the shared benchmark GPU lock for one complete driver operation."""
    @wraps(function)
    def wrapped(*args: P.args, **kwargs: P.kwargs) -> R:
        with gpu_benchmark_lock():
            return function(*args, **kwargs)

    return wrapped


def token() -> str:
    p = Path(os.environ.get("FTS_TOKEN_FILE", ROOT / ".tmp" / "bench" / "token"))
    return p.read_text().strip()


def client(timeout: float = 120.0) -> httpx.Client:
    return httpx.Client(base_url=BASE, headers={"Authorization": f"Bearer {token()}"}, timeout=timeout)


def check(r: httpx.Response) -> Any:
    if r.status_code >= 400:
        raise RuntimeError(f"{r.request.method} {r.request.url.path} -> {r.status_code}: {r.text[:500]}")
    return r.json() if r.content else None


def post_when_free(c: httpx.Client, path: str, *, every: float = 30.0, timeout: float = 4 * 3600, **kw) -> Any:
    """POST, waiting while the app answers 409 because the GPU (or a build) is busy: training and index builds
    take the same GPU lease, so two benchmark drivers can share the card one job at a time."""
    t0 = time.time()
    while True:
        r = c.post(path, **kw)
        if r.status_code == 409 and time.time() - t0 < timeout:
            time.sleep(every)
            continue
        return check(r)


def project_by_name(c: httpx.Client, name: str) -> dict | None:
    for p in check(c.get("/api/projects")):
        if (p.get("name") or "").strip().lower() == name.lower():
            return p
    return None


def ensure_project(c: httpx.Client, name: str, description: str = "") -> dict:
    p = project_by_name(c, name)
    if p:
        return p
    return check(c.post("/api/projects", json={"name": name, "description": description}))


def load_jsonl(p: Path) -> list[dict]:
    return [json.loads(x) for x in p.read_text().splitlines() if x.strip()]


def norm(s: str) -> str:
    """Normalize text for case/whitespace-insensitive marker substring checks."""
    s = unicodedata.normalize("NFKC", s or "").lower()
    return " ".join(s.split())


def contains(hay_norm: str, needle: str) -> bool:
    """Return whether a normalized marker occurs as a substring (not a field-boundary match)."""
    return norm(needle) in hay_norm


def contains_whole_marker(hay_norm: str, needle: str) -> bool:
    """Match a normalized marker outside word and hyphenated identifier characters."""
    normalized = norm(needle)
    if not normalized:
        return False
    return re.search(rf"(?<![\w-]){re.escape(normalized)}(?![\w-])", hay_norm) is not None


def wait(fn, *, every: float = 5.0, timeout: float = 3600.0, what: str = "") -> Any:
    t0 = time.time()
    while True:
        v = fn()
        if v:
            return v
        if time.time() - t0 > timeout:
            raise TimeoutError(f"timed out waiting for {what}")
        time.sleep(every)


def write_json(p: Path, obj: Any) -> None:
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_name(p.name + ".tmp")
    tmp.write_text(json.dumps(obj, indent=2, ensure_ascii=False))
    tmp.replace(p)
