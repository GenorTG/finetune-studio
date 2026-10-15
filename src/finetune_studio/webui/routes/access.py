"""Sign-in, sign-out and Settings -> Security (password, API tokens, sessions, listen address).

The middleware (``access.middleware``) decides who gets in; these routes change the state it reads.
"""
from __future__ import annotations

import asyncio
import math
import os
import time
from pathlib import Path
from typing import Any

from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import HTMLResponse, JSONResponse, RedirectResponse, Response
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel, Field

from finetune_studio.access import netinfo, server, sessions, store, tailscale, throttle
from finetune_studio.access.middleware import (
    exposed_without_password,
    listening,
    session_token,
)
from finetune_studio.access.passwords import MIN_LENGTH

router = APIRouter()
templates = Jinja2Templates(directory=str(Path(__file__).parent.parent / "templates"))


def _client(request: Request) -> str:
    return request.client.host if request.client else "unknown"


def _safe_next(target: str | None) -> str:
    """Only same-site paths: '/x' yes, '//evil.example' and 'https://...' no."""
    if not target or not target.startswith("/") or target.startswith("//") or "\\" in target:
        return "/"
    return target


def _set_cookie(response: Response, request: Request, token: str) -> None:
    response.set_cookie(sessions.COOKIE, token, max_age=sessions.ABSOLUTE_TTL_S, httponly=True, samesite="lax",
                        secure=request.url.scheme == "https", path="/")


def _require_password(cfg: store.AuthConfig, given: str | None) -> None:
    if cfg.enabled and not store.check_password(given or ""):
        raise HTTPException(status_code=403, detail="the current password is wrong")


# ── sign in / out ───────────────────────────────────────────────────────

@router.get("/login", response_class=HTMLResponse)
async def login_page(request: Request, next: str | None = None):
    if not store.load().enabled:
        return RedirectResponse(_safe_next(next), status_code=303)
    return templates.TemplateResponse(request, "login.html", {"next": _safe_next(next)})


class LoginBody(BaseModel):
    password: str = Field(min_length=1, max_length=1024)
    next: str | None = None


@router.post("/api/auth/login")
async def login(body: LoginBody, request: Request):
    cfg = store.load()
    if not cfg.enabled:
        raise HTTPException(status_code=409, detail="no password is set: this instance does not ask anyone to sign in")
    client = _client(request)
    wait = throttle.retry_after(client)
    if wait > 0:
        return JSONResponse({"detail": f"too many wrong passwords; try again in {math.ceil(wait)} s",
                             "code": "throttled"}, status_code=429, headers={"Retry-After": str(math.ceil(wait))})
    if not store.check_password(body.password):
        throttle.failed(client)
        raise HTTPException(status_code=401, detail="wrong password")
    throttle.succeeded(client)
    token = sessions.create(user_agent=request.headers.get("user-agent", ""), client=client)
    resp = JSONResponse({"next": _safe_next(body.next)})
    _set_cookie(resp, request, token)
    return resp


@router.post("/api/auth/logout")
async def logout(request: Request):
    token = request.cookies.get(sessions.COOKIE)
    if token:
        sessions.revoke(token)
    resp = JSONResponse({"signed_out": True})
    resp.delete_cookie(sessions.COOKIE, path="/")
    return resp


@router.get("/api/auth/state")
async def auth_state(request: Request) -> dict[str, Any]:
    cfg = store.load()
    return {"mode": cfg.mode, "signed_in": cfg.enabled and sessions.validate(request.cookies.get(sessions.COOKIE)),
            "exposed_without_password": exposed_without_password()}


# ── Settings -> Security ────────────────────────────────────────────────

async def _security_state() -> dict[str, Any]:
    cfg = store.load()
    addr = server.load()
    bound = listening()
    listening_port = int(os.environ.get("FTS_LISTEN_PORT") or addr.port)
    tailnet = await asyncio.to_thread(tailscale.detect_cached)
    lan = await asyncio.to_thread(netinfo.lan_addresses)
    wildcard = any(server.is_wildcard(b) for b in bound)
    return {
        "mode": cfg.mode,
        "password_set": cfg.enabled,
        "password_changed_at": cfg.password_changed_at,
        "password_min_length": MIN_LENGTH,
        "tokens": [t.public() for t in cfg.tokens],
        "sessions": sessions.count(),
        # Settings -> Access & network: one switch per kind of device, plus custom addresses.
        "reach": {
            "lan": {"on": addr.lan, "listening": addr.lan and wildcard, "addresses": [a.public() for a in lan]},
            "tailscale": {"on": addr.tailscale, "detected": tailnet.public() if tailnet else None,
                          "ip": addr.tailscale_ip, "names": list(addr.tailscale_names),
                          "listening": addr.tailscale and (wildcard or addr.tailscale_ip in bound)},
            "custom": {"addresses": list(addr.custom),
                       "listening": [c for c in addr.custom if c in bound or wildcard]},
            "now": server.effective(bound, addr),
            "saved": server.describe(addr),
        },
        "server": {"port": addr.port, "allowed_hosts": list(addr.allowed_hosts),
                   "binds": list(addr.binds()), "listening": list(bound), "listening_port": listening_port,
                   "loopback_only": addr.loopback_only,
                   # Saved but not in use yet: switching ON or a new port applies when the web UI restarts.
                   "restart_needed": (tuple(bound), listening_port) != (addr.binds(), addr.port),
                   "listen_error": os.environ.get("FTS_LISTEN_ERROR", "")},
        "exposed_without_password": exposed_without_password(),
    }


@router.get("/api/settings/security")
async def security_state(request: Request) -> dict[str, Any]:
    return await _security_state()


class PasswordBody(BaseModel):
    new_password: str = Field(max_length=1024)
    current_password: str | None = None


@router.put("/api/settings/security/password")
async def set_password(body: PasswordBody, request: Request):
    cfg = store.load()
    _require_password(cfg, body.current_password)
    try:
        store.set_password(body.new_password)
    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e)) from e
    # Everyone else signs in again with the new password; this browser stays in with a fresh session.
    sessions.revoke_all()
    token = sessions.create(user_agent=request.headers.get("user-agent", ""), client=_client(request))
    resp = JSONResponse(await _security_state())
    _set_cookie(resp, request, token)
    return resp


class ConfirmBody(BaseModel):
    current_password: str | None = None


@router.post("/api/settings/security/disable")
async def disable_password(body: ConfirmBody, request: Request):
    cfg = store.load()
    if not cfg.enabled:
        raise HTTPException(status_code=409, detail="no password is set")
    _require_password(cfg, body.current_password)
    store.disable()
    sessions.revoke_all()
    resp = JSONResponse(await _security_state())
    resp.delete_cookie(sessions.COOKIE, path="/")
    return resp


class TokenBody(BaseModel):
    name: str = Field(min_length=1, max_length=60)


@router.post("/api/settings/security/tokens", status_code=201)
async def create_token(body: TokenBody) -> dict[str, Any]:
    tok, plain = store.create_token(body.name)
    return {"token": plain, **tok.public(), "note": "shown once: copy it now"}


@router.delete("/api/settings/security/tokens/{token_id}", status_code=204)
async def revoke_token(token_id: str) -> Response:
    if not store.revoke_token(token_id):
        raise HTTPException(status_code=404, detail=f"no API token {token_id!r}")
    return Response(status_code=204)


@router.delete("/api/settings/security/sessions")
async def clear_sessions(request: Request) -> dict[str, Any]:
    removed = sessions.revoke_all(except_token=session_token({"cookie": request.headers.get("cookie", "")}))
    return {"signed_out_elsewhere": removed}


class ServerBody(BaseModel):
    """Every field optional: None = keep as saved. ``host`` is the pre-switches single address."""
    port: int | None = None
    lan: bool | None = None
    tailscale: bool | None = None
    custom: list[str] | None = None
    allowed_hosts: list[str] | None = None
    host: str | None = None
    confirm_no_password: bool = False


@router.put("/api/settings/security/server")
async def set_server(body: ServerBody, request: Request) -> dict[str, Any]:
    saved = server.load()
    tailnet = await asyncio.to_thread(tailscale.detect) if body.tailscale else None
    if body.tailscale and tailnet is None and not saved.tailscale_ip:
        raise HTTPException(status_code=409, detail=(
            "Tailscale is not detected on this computer (not installed, logged out or down): "
            "start it (tailscale up) and try again"))
    try:
        base = saved
        if body.host:
            legacy = server.from_host(body.host, body.port or saved.port, saved.allowed_hosts)
            base = server.change(saved, lan=legacy.lan, tailscale=legacy.tailscale, custom=legacy.custom)
        addr = server.change(base, port=body.port, lan=body.lan, tailscale=body.tailscale, custom=body.custom,
                             allowed_hosts=body.allowed_hosts, tailnet=tailnet)
    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e)) from e
    if not addr.loopback_only and not store.load().enabled and not body.confirm_no_password:
        raise HTTPException(status_code=409, detail=(
            f"this lets {server.describe(addr).removeprefix('this computer + ')} reach this instance, and no "
            "password is set: anyone there could run training, delete projects or start an update. "
            "Set a password first, or confirm."))
    server.save(addr)
    state = await _security_state()
    state["saved_at"] = time.time()
    return state
