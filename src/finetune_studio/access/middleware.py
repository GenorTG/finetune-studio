"""ASGI middleware: sign-in when a password is set, plus browser protections that apply even without one.

Always:
- a state-changing request (POST/PUT/PATCH/DELETE) whose ``Origin`` is another site is refused (403): otherwise any
  web page you visit could POST to ``localhost:7860`` (start an update, delete a project);
- while listening on loopback (plus Tailscale), the ``Host`` must be a loopback name, in ``server.json``
  ``allowed_hosts``, or this machine's Tailscale IP / MagicDNS name: blocks DNS-rebinding pages posing as a local site.
With a password (``auth.json`` mode ``password``): every route except the sign-in page, static files and
``/api/health`` needs a session cookie or ``Authorization: Bearer fts_...``. Pages redirect to ``/login``, API calls
answer 401 JSON. Cookie-authenticated writes must carry a same-origin ``Origin`` or ``Referer`` (CSRF).
"""
from __future__ import annotations

import json
import logging
import os
from http.cookies import SimpleCookie
from typing import Any
from urllib.parse import quote, urlsplit

from finetune_studio.access import server, sessions, store
from finetune_studio.access.paths import auth_file, server_file

_log = logging.getLogger(__name__)

PUBLIC_PREFIXES = ("/static/", "/login", "/api/auth/login", "/api/auth/state", "/api/health", "/favicon.ico")
UNSAFE = {"POST", "PUT", "PATCH", "DELETE"}
LOOPBACK_NAMES = {"localhost", "127.0.0.1", "::1", "[::1]"}


def listen_host() -> str:
    """The address this process listens on: the supervisor passes it; standalone runs fall back to server.json."""
    return os.environ.get("FTS_LISTEN_HOST") or _address().host


def listen_extra() -> tuple[str, ...]:
    """Extra addresses the supervisor bound (the Tailscale IP); empty for a standalone run."""
    return tuple(h for h in os.environ.get("FTS_LISTEN_EXTRA", "").split(",") if h)


def exposed_without_password() -> bool:
    """True when other machines can reach this instance and nothing asks them for a password."""
    try:
        reachable = not server.is_loopback(listen_host()) or bool(listen_extra())
        return reachable and not _config().enabled
    except RuntimeError:
        return False


_cache: dict[str, Any] = {"mtime": None, "cfg": None}
_addr_cache: dict[str, Any] = {"mtime": None, "addr": None}


def _address() -> server.ServerAddress:
    f = server_file()
    mtime = f.stat().st_mtime_ns if f.exists() else None
    if _addr_cache["addr"] is None or mtime != _addr_cache["mtime"]:
        _addr_cache["addr"], _addr_cache["mtime"] = server.load(), mtime
    return _addr_cache["addr"]


def _config() -> store.AuthConfig:
    """auth.json, re-read only when it changes (Settings / the CLI write it while the app runs)."""
    f = auth_file()
    mtime = f.stat().st_mtime_ns if f.exists() else None
    if _cache["cfg"] is None or mtime != _cache["mtime"]:
        _cache["cfg"], _cache["mtime"] = store.load(), mtime
    return _cache["cfg"]


def _headers(scope: dict[str, Any]) -> dict[str, str]:
    return {k.decode("latin-1").lower(): v.decode("latin-1") for k, v in scope.get("headers", [])}


def _host_only(value: str) -> str:
    v = value.strip().lower()
    if v.startswith("["):
        return v[: v.find("]") + 1]
    return v.rsplit(":", 1)[0] if v.count(":") == 1 else v


def _same_origin(origin_or_referer: str, host_header: str) -> bool:
    parts = urlsplit(origin_or_referer)
    return bool(parts.netloc) and parts.netloc.lower() == host_header.strip().lower()


def session_token(headers: dict[str, str]) -> str | None:
    raw = headers.get("cookie")
    if not raw:
        return None
    jar = SimpleCookie()
    try:
        jar.load(raw)
    except Exception:  # noqa: BLE001 — a malformed Cookie header is simply "no session"
        return None
    m = jar.get(sessions.COOKIE)
    return m.value if m else None


class AccessMiddleware:
    def __init__(self, app: Any, trusted_origins: list[str] | None = None) -> None:
        self.app = app
        # Origins configured for CORS (Settings -> hosting) may write cross-site; nothing else may.
        self.trusted_origins = {o.rstrip("/").lower() for o in trusted_origins or []}

    async def __call__(self, scope: dict[str, Any], receive: Any, send: Any) -> None:
        if scope["type"] not in ("http", "websocket"):
            await self.app(scope, receive, send)
            return
        path: str = scope.get("path", "")
        method: str = scope.get("method", "GET").upper()
        h = _headers(scope)
        host = h.get("host", "")

        if method == "OPTIONS":                     # CORS preflight: no credentials, changes nothing
            await self.app(scope, receive, send)
            return
        refusal = self._browser_guard(method, h, host, self.trusted_origins)
        if refusal:
            await _reply(scope, send, 403, refusal, "request_refused")
            return

        try:
            cfg = _config()
        except RuntimeError as e:
            _log.error("access config unreadable, refusing requests: %s", e)
            await _reply(scope, send, 503, str(e), "access_config_broken")
            return
        if not cfg.enabled or path.startswith(PUBLIC_PREFIXES):
            await self.app(scope, receive, send)
            return

        auth = h.get("authorization", "")
        if auth.lower().startswith("bearer "):
            if store.verify_token(auth[7:].strip(), cfg):
                await self.app(scope, receive, send)
                return
            await _reply(scope, send, 401, "invalid API token", "auth_invalid_token")
            return

        if sessions.validate(session_token(h)):
            if method in UNSAFE:
                src = h.get("origin") or h.get("referer") or ""
                if not (_same_origin(src, host) or src.rstrip("/").lower() in self.trusted_origins):
                    await _reply(scope, send, 403, "cross-site request refused (missing or foreign Origin)", "csrf")
                    return
            await self.app(scope, receive, send)
            return

        wants_html = method == "GET" and not path.startswith("/api/") and "text/html" in h.get("accept", "")
        if wants_html and scope["type"] == "http":
            qs = scope.get("query_string", b"").decode("latin-1")
            target = path + (f"?{qs}" if qs else "")
            await _redirect(send, "/login?next=" + quote(target, safe=""))
            return
        await _reply(scope, send, 401, "sign in required", "auth_required")

    @staticmethod
    def _browser_guard(method: str, h: dict[str, str], host: str, trusted: set[str]) -> str | None:
        origin = h.get("origin")
        if (method in UNSAFE and origin and not _same_origin(origin, host)
                and origin.rstrip("/").lower() not in trusted):
            return f"cross-site request from {origin} refused"
        try:
            bind = listen_host()
            addr = _address()
        except ValueError:
            return None
        if server.is_loopback(bind) and host:
            name = _host_only(host)
            tailnet = (addr.tailscale_ip, *addr.tailscale_names) if addr.tailscale else ()
            if name not in LOOPBACK_NAMES and name not in addr.allowed_hosts and name not in tailnet:
                return (f"host name {name!r} is not allowed while Finetune Studio listens on this computer only "
                        f"(add it under Settings -> Security -> allowed host names)")
        return None


async def _reply(scope: dict[str, Any], send: Any, status: int, detail: str, code: str) -> None:
    if scope["type"] == "websocket":
        await send({"type": "websocket.close", "code": 4401 if status == 401 else 4403})
        return
    body = json.dumps({"detail": detail, "code": code}).encode()
    headers = [(b"content-type", b"application/json"), (b"content-length", str(len(body)).encode()),
               (b"cache-control", b"no-store")]
    if status == 401:
        headers.append((b"x-fts-auth", b"sign-in"))       # static/js/access.js sends the page to /login
    await send({"type": "http.response.start", "status": status, "headers": headers})
    await send({"type": "http.response.body", "body": body})


async def _redirect(send: Any, location: str) -> None:
    await send({"type": "http.response.start", "status": 303,
                "headers": [(b"location", location.encode("latin-1")), (b"content-length", b"0"),
                            (b"cache-control", b"no-store")]})
    await send({"type": "http.response.body", "body": b""})
