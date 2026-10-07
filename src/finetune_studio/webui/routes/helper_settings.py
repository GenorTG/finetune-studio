"""Helper-model settings: seat the local GGUF or an API provider as the helper.

The helper does the tedious model work (mining Q&A, suite generation, preference pairs, the Guide).
Settings → "Helper model" picks who holds that seat: the local GGUF on this machine (default) or an
OpenAI-compatible API provider the user connected. The API key is write-only: it is stored in the
local provider DB (never in the repo) and no endpoint returns it.
"""
from __future__ import annotations

import asyncio
import logging
import time
from typing import Any
from urllib.parse import urlparse

import requests
from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import JSONResponse

from finetune_studio.models.helper import (
    API_HELPER_PROVIDER_ID,
    DEFAULT_HELPER_PROVIDER_ID,
    get_configured_helper_provider,
    get_helper_provider_id,
    set_helper_provider_id,
)
from finetune_studio.models.manager import get_manager
from finetune_studio.models.providers import (
    PROVIDER_PRESETS,
    OpenAICompatProvider,
    ProviderConfig,
    ProviderError,
)
from finetune_studio.webui.engine_guard import ENGINE_LOCK

router = APIRouter()
log = logging.getLogger(__name__)

REASONING_EFFORTS = ("", "none", "minimal", "low", "medium", "high")
_API_PRESET_IDS = ("opencode-go", "openai", "openrouter", "custom")


def _presets() -> list[dict[str, Any]]:
    out = []
    for p in PROVIDER_PRESETS:
        if p["id"] in _API_PRESET_IDS:
            out.append({
                "id": p["id"], "label": p["name"], "base_url": p["base_url"], "model": p["model_id"],
                "headers": sorted((p.get("extra") or {}).get("headers") or {}),
            })
    return out


def _public_state() -> dict[str, Any]:
    mgr = get_manager()
    seat = get_helper_provider_id()
    api = mgr.get_provider(API_HELPER_PROVIDER_ID)
    local = mgr.get_provider(DEFAULT_HELPER_PROVIDER_ID) or {}
    active = mgr.active() or {}
    extra = (api or {}).get("extra") or {}
    return {
        "seat": "api" if seat == API_HELPER_PROVIDER_ID else "local",
        "seat_provider_id": seat,
        "seat_label": (get_configured_helper_provider() or {}).get("label", ""),
        "local": {"provider_id": DEFAULT_HELPER_PROVIDER_ID, "label": local.get("label", ""),
                  "model_path": local.get("model_id", "")},
        "api": {
            "configured": bool(api),
            "base_url": (api or {}).get("base_url", ""),
            "model": (api or {}).get("model_id", ""),
            "key_set": bool((api or {}).get("api_key_set")),
            "reasoning_effort": str((extra.get("body") or {}).get("reasoning_effort") or ""),
            "preset": str(extra.get("preset") or ""),
        },
        "presets": _presets(),
        "reasoning_efforts": [e for e in REASONING_EFFORTS if e],
        "loaded_provider_id": active.get("id") or "",
    }


@router.get("/api/settings/helper")
async def get_helper_settings():
    return await asyncio.to_thread(_public_state)


def _validate_url(url: str) -> str:
    parsed = urlparse(url)
    if parsed.scheme not in ("http", "https") or not parsed.netloc:
        raise HTTPException(status_code=400, detail="API base URL must start with http:// or https://")
    return url.rstrip("/")


def _upsert_api_row(body: dict[str, Any]) -> None:
    """Create/update the ``api-helper`` row from a settings body; a blank key keeps the stored one."""
    mgr = get_manager()
    existing = mgr.get_provider(API_HELPER_PROVIDER_ID)
    base_url = _validate_url(str(body.get("base_url") or (existing or {}).get("base_url") or "").strip())
    model = str(body.get("model") or (existing or {}).get("model_id") or "").strip()
    if not model:
        raise HTTPException(status_code=400, detail="choose a model name for the API provider")
    effort = str(body.get("reasoning_effort") or "").strip().lower()
    if effort and effort not in REASONING_EFFORTS:
        raise HTTPException(status_code=400, detail=f"reasoning_effort must be one of {REASONING_EFFORTS[1:]}")
    preset_id = str(body.get("preset") or (existing or {}).get("extra", {}).get("preset") or "custom")
    preset = next((p for p in PROVIDER_PRESETS if p["id"] == preset_id), None)
    extra: dict[str, Any] = {"preset": preset_id}
    headers = (preset or {}).get("extra", {}).get("headers")
    if headers:
        extra["headers"] = dict(headers)
    if effort:
        extra["body"] = {"reasoning_effort": effort}
    fields: dict[str, Any] = {
        "id": API_HELPER_PROVIDER_ID, "name": "Helper · API", "kind": "openai_compat",
        "model_id": model, "base_url": base_url, "extra": extra,
    }
    key = str(body.get("api_key") or "").strip()
    if key:
        fields["api_key"] = key
    elif body.get("clear_key"):
        fields["api_key"] = ""
    mgr.upsert_provider(**fields)
    if mgr.active() and mgr.active().get("id") == API_HELPER_PROVIDER_ID:
        mgr.unload()  # settings changed: the next call reloads with the new url/model/key


@router.put("/api/settings/helper")
async def put_helper_settings(request: Request):
    try:
        body = await request.json()
    except ValueError:
        raise HTTPException(status_code=400, detail="invalid JSON") from None
    if not isinstance(body, dict):
        raise HTTPException(status_code=400, detail="expected JSON object")
    seat = str(body.get("seat") or "").strip().lower()
    if seat not in ("local", "api"):
        raise HTTPException(status_code=400, detail="seat must be 'local' or 'api'")

    def _apply() -> None:
        mgr = get_manager()
        if seat == "api" or body.get("base_url") or body.get("model"):
            _upsert_api_row(body)
        if seat == "api":
            row = mgr.get_provider(API_HELPER_PROVIDER_ID)
            if not row or not row.get("api_key_set"):
                raise HTTPException(status_code=400, detail="enter the API key before seating the API helper")
            set_helper_provider_id(API_HELPER_PROVIDER_ID)
        else:
            set_helper_provider_id(DEFAULT_HELPER_PROVIDER_ID)

    # Moving the seat away from the local GGUF frees its VRAM; the API helper itself loads instantly.
    async with ENGINE_LOCK:
        before = get_helper_provider_id()
        await asyncio.to_thread(_apply)
        after = get_helper_provider_id()
        if before != after:
            from finetune_studio.models.llama_loader import unload_all_models
            await asyncio.to_thread(unload_all_models)
            if after == API_HELPER_PROVIDER_ID:
                await asyncio.to_thread(get_manager().load, API_HELPER_PROVIDER_ID)
    return await asyncio.to_thread(_public_state)


def _probe_config(body: dict[str, Any]) -> ProviderConfig:
    """Config to test: typed-in values over the stored row (the stored key when none is typed)."""
    mgr = get_manager()
    row = mgr.get_provider(API_HELPER_PROVIDER_ID) or {}
    preset = next((p for p in PROVIDER_PRESETS if p["id"] == (body.get("preset") or (row.get("extra") or {}).get("preset"))), None)
    extra: dict[str, Any] = {"retries": 1, "timeout": 60}
    headers = (preset or {}).get("extra", {}).get("headers") or (row.get("extra") or {}).get("headers")
    if headers:
        extra["headers"] = dict(headers)
    effort = str(body.get("reasoning_effort") or "").strip().lower()
    if effort:
        extra["body"] = {"reasoning_effort": effort}
    return ProviderConfig(
        id=API_HELPER_PROVIDER_ID, name="Helper · API", kind="openai_compat",
        model_id=str(body.get("model") or row.get("model_id") or "").strip(),
        base_url=_validate_url(str(body.get("base_url") or row.get("base_url") or "").strip()),
        api_key=str(body.get("api_key") or "").strip() or mgr._api_key_for(API_HELPER_PROVIDER_ID),
        extra=extra,
    )


@router.post("/api/settings/helper/test")
async def test_helper_api(request: Request):
    """One real chat call with the typed-in (or stored) settings: proves URL, key, model and latency."""
    try:
        body = await request.json()
    except ValueError:
        body = {}
    cfg = _probe_config(body if isinstance(body, dict) else {})
    if not cfg.model_id or not cfg.api_key:
        return JSONResponse({"ok": False, "error": "enter the model name and API key first"}, status_code=400)
    provider = OpenAICompatProvider(cfg)

    def _ping() -> dict[str, Any]:
        t0 = time.perf_counter()
        try:
            reply = provider.chat([{"role": "user", "content": "Reply with exactly: pong"}], max_tokens=512, temperature=0)
        except ProviderError as exc:
            return {"ok": False, "error": str(exc), "latency_ms": int((time.perf_counter() - t0) * 1000)}
        finally:
            provider.unload()
        return {"ok": True, "reply": reply[:80], "latency_ms": int((time.perf_counter() - t0) * 1000)}

    result = await asyncio.to_thread(_ping)
    return JSONResponse(result, status_code=200 if result["ok"] else 502)


@router.post("/api/settings/helper/models")
async def list_helper_api_models(request: Request):
    """Model ids the provider offers (``GET {base}/models``), so the user picks instead of typing."""
    try:
        body = await request.json()
    except ValueError:
        body = {}
    cfg = _probe_config(body if isinstance(body, dict) else {})
    if not cfg.api_key:
        return JSONResponse({"ok": False, "error": "enter the API key first"}, status_code=400)
    headers = {"Authorization": f"Bearer {cfg.api_key}"}

    def _fetch() -> dict[str, Any]:
        try:
            r = requests.get(cfg.base_url + "/models", headers=headers, timeout=20)
        except requests.RequestException as exc:
            return {"ok": False, "error": f"{type(exc).__name__} calling {cfg.base_url}/models"}
        if r.status_code >= 400:
            return {"ok": False, "error": f"HTTP {r.status_code}: {' '.join(r.text.split())[:200]}"}
        try:
            data = r.json().get("data") or []
            return {"ok": True, "models": sorted(str(m["id"]) for m in data if isinstance(m, dict) and m.get("id"))}
        except (ValueError, AttributeError):
            return {"ok": False, "error": "the provider's /models answer was not the OpenAI list shape"}

    result = await asyncio.to_thread(_fetch)
    return JSONResponse(result, status_code=200 if result["ok"] else 502)
