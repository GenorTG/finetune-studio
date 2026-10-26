"""Settings -> AI assistant: connections, the main assistant and the jobs it does (``models/assistant.py``).

Connections are independent of each other: the local model (``local-default``, managed by ``helper_settings``) and any
number of OpenAI-compatible API connections, each with its own key. A key is write-only: it is stored encrypted
(``access.vault``) and no endpoint returns it. Removing a connection sends the jobs that used it back to the main
assistant; the main assistant itself cannot be removed until another connection takes its place.
"""
from __future__ import annotations

import asyncio
import logging
import re
import time
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

import requests
from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import JSONResponse

from finetune_studio.data.mining.scheduler import (
    DEFAULT_API_CONCURRENCY,
    MAX_API_CONCURRENCY,
)
from finetune_studio.models import assistant
from finetune_studio.models.helper import (
    DEFAULT_HELPER_PROVIDER_ID,
    recommended_helper_state,
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


async def _json(request: Request) -> dict[str, Any]:
    try:
        body = await request.json()
    except ValueError:
        raise HTTPException(status_code=400, detail="invalid JSON") from None
    if not isinstance(body, dict):
        raise HTTPException(status_code=400, detail="expected JSON object")
    return body


def _presets() -> list[dict[str, Any]]:
    return [{"id": p["id"], "label": p["name"], "base_url": p["base_url"], "model": p["model_id"]}
            for p in PROVIDER_PRESETS if p["id"] in _API_PRESET_IDS]


def _concurrency(extra: dict[str, Any]) -> int:
    try:
        return max(1, min(MAX_API_CONCURRENCY, int(extra.get("concurrency") or DEFAULT_API_CONCURRENCY)))
    except (TypeError, ValueError):
        return DEFAULT_API_CONCURRENCY


def _local_candidates() -> list[dict[str, Any]]:
    """Installed chat GGUFs that could be the local model (vision projectors are not models)."""
    from finetune_studio.webui.app import discovered_models

    out = []
    for m in discovered_models:
        name = Path(str(getattr(m, "path", ""))).name
        if str(getattr(m, "format", "")).lower() == "gguf" and not name.lower().startswith("mmproj"):
            out.append({"path": str(m.path), "name": name, "size_gb": getattr(m, "size_gb", None)})
    return sorted(out, key=lambda c: c["name"].lower())


def _vision_candidates() -> list[dict[str, str]]:
    from finetune_studio.webui.routes.vision_settings import (
        _local_candidates as vision_local,
    )

    return vision_local()


def _connection(row: dict[str, Any], seat: str, used_by: dict[str, list[str]]) -> dict[str, Any]:
    extra = row.get("extra") or {}
    api = row.get("kind") == "openai_compat"
    out: dict[str, Any] = {
        "id": row["id"], "kind": row.get("kind", ""), "name": row.get("name") or row["id"],
        "label": assistant.connection_label(row), "model": row.get("model_id", ""),
        "is_assistant": row["id"] == seat, "used_by": used_by.get(row["id"], []),
    }
    if api:
        out.update({"base_url": row.get("base_url", ""), "key_set": bool(row.get("api_key_set")),
                    "key_state": row.get("api_key_state") or "unset", "preset": str(extra.get("preset") or "custom"),
                    "reasoning_effort": str((extra.get("body") or {}).get("reasoning_effort") or ""),
                    "concurrency": _concurrency(extra)})
    else:
        path = str(row.get("model_id") or "")
        out.update({"path": path, "present": bool(path) and Path(path).expanduser().is_file()})
    return out


def _state() -> dict[str, Any]:
    mgr = get_manager()
    seat = assistant.assistant_provider_id()
    jobs = assistant.jobs_state()
    used_by: dict[str, list[str]] = {}
    for j in jobs:
        if j["setting"] not in ("", assistant.OFF, "local-file"):
            used_by.setdefault(j["provider_id"], []).append(j["label"])
    # The local model is the one ``local-default`` row; other local rows (the alternate helper seeded by older
    # versions) only show up when something uses them.
    rows = [r for r in mgr.list_providers()
            if r.get("kind") == "openai_compat" or r["id"] == DEFAULT_HELPER_PROVIDER_ID or r["id"] in (seat, *used_by)]
    rows.sort(key=lambda r: (r["id"] != DEFAULT_HELPER_PROVIDER_ID, float(r.get("created_at") or 0)))
    return {
        "assistant_provider_id": seat,
        "assistant_label": assistant.connection_label(mgr.get_provider(seat)),
        "connections": [_connection(r, seat, used_by) for r in rows],
        "local": {"provider_id": DEFAULT_HELPER_PROVIDER_ID, "recommended": recommended_helper_state(),
                  "candidates": _local_candidates()},
        "jobs": jobs,
        "vision_candidates": _vision_candidates(),
        "presets": _presets(),
        "reasoning_efforts": [e for e in REASONING_EFFORTS if e],
        "max_concurrency": MAX_API_CONCURRENCY,
        "loaded_provider_id": (mgr.active() or {}).get("id") or "",
    }


@router.get("/api/settings/assistant")
async def get_assistant_settings():
    return await asyncio.to_thread(_state)


@router.put("/api/settings/assistant")
async def put_assistant(request: Request):
    """Make another connection the main assistant. A local model that stops being the assistant is unloaded."""
    body = await _json(request)
    pid = str(body.get("provider_id") or "").strip()
    mgr = get_manager()
    row = mgr.get_provider(pid) if pid else None
    if row is None:
        raise HTTPException(status_code=400, detail=f"unknown connection {pid!r}")
    if row.get("kind") == "openai_compat" and not row.get("api_key_set"):
        raise HTTPException(status_code=400, detail=f"{row.get('name') or pid} has no saved API key yet")
    async with ENGINE_LOCK:
        before = assistant.assistant_provider_id()
        await asyncio.to_thread(assistant.set_assistant_provider_id, pid)
        if before != pid and (mgr.active() or {}).get("id") == before:
            from finetune_studio.models.llama_loader import unload_all_models
            await asyncio.to_thread(unload_all_models)
    return await asyncio.to_thread(_state)


@router.put("/api/settings/assistant/jobs/{job_id}")
async def put_job(job_id: str, request: Request):
    """A job's connection (``""`` = main assistant, ``"off"`` for vision, a connection id, or for vision
    ``"local-file"`` + ``local_model``) and/or its extra instructions."""
    body = await _json(request)
    if job_id not in assistant.JOB_IDS:
        raise HTTPException(status_code=404, detail=f"unknown job {job_id!r}")
    provider = body.get("provider_id")
    instructions = body.get("instructions")
    if provider is not None and not isinstance(provider, str):
        raise HTTPException(status_code=400, detail="provider_id must be a string")
    if instructions is not None and not isinstance(instructions, str):
        raise HTTPException(status_code=400, detail="instructions must be a string")

    def _apply() -> dict[str, Any]:
        try:
            assistant.set_job(job_id, provider_id=provider, instructions=instructions,
                              local_model=str(body.get("local_model") or ""))
        except ValueError as e:
            raise HTTPException(status_code=400, detail=str(e)) from None
        return _state()

    return await asyncio.to_thread(_apply)


# ── API connections ─────────────────────────────────────────────────────────

def _validate_url(url: str) -> str:
    parsed = urlparse(url)
    if parsed.scheme not in ("http", "https") or not parsed.netloc:
        raise HTTPException(status_code=400, detail="API base URL must start with http:// or https://")
    return url.rstrip("/")


def _new_id(name: str) -> str:
    slug = re.sub(r"[^a-z0-9]+", "-", name.lower()).strip("-")[:32] or "connection"
    mgr = get_manager()
    pid, n = f"api-{slug}", 2
    while mgr.get_provider(pid) is not None:
        pid, n = f"api-{slug}-{n}", n + 1
    return pid


def _upsert_connection(pid: str | None, body: dict[str, Any]) -> str:
    """Create (``pid`` None) or update an API connection; a blank key keeps the stored one, ``clear_key`` removes it.
    A field the body omits keeps its stored value."""
    mgr = get_manager()
    existing = mgr.get_provider(pid) if pid else None
    if pid and existing is None:
        raise HTTPException(status_code=404, detail=f"unknown connection {pid!r}")
    if existing is not None and existing.get("kind") != "openai_compat":
        raise HTTPException(status_code=400, detail="the local model is set up with its own controls, not as an API")
    old_extra = dict((existing or {}).get("extra") or {})
    name = str(body.get("name") or (existing or {}).get("name") or "").strip()
    if not name:
        raise HTTPException(status_code=400, detail="give the connection a name")
    base_url = _validate_url(str(body.get("base_url") or (existing or {}).get("base_url") or "").strip())
    model = str(body.get("model") or (existing or {}).get("model_id") or "").strip()
    if not model:
        raise HTTPException(status_code=400, detail="choose a model name for the connection")
    stored_effort = str((old_extra.get("body") or {}).get("reasoning_effort") or "")
    effort = str(body.get("reasoning_effort", stored_effort) or "").strip().lower()
    if effort not in REASONING_EFFORTS:
        raise HTTPException(status_code=400, detail=f"reasoning_effort must be one of {REASONING_EFFORTS[1:]}")
    raw_conc = body.get("concurrency", old_extra.get("concurrency"))
    try:
        concurrency = int(raw_conc) if raw_conc not in (None, "") else DEFAULT_API_CONCURRENCY
    except (TypeError, ValueError):
        raise HTTPException(status_code=400, detail="concurrency must be a whole number") from None
    if not 1 <= concurrency <= MAX_API_CONCURRENCY:
        raise HTTPException(status_code=400, detail=f"concurrency must be between 1 and {MAX_API_CONCURRENCY}")
    preset_id = str(body.get("preset") or old_extra.get("preset") or "custom")
    preset = next((p for p in PROVIDER_PRESETS if p["id"] == preset_id), None)
    extra: dict[str, Any] = {"preset": preset_id, "concurrency": concurrency}
    headers = (preset or {}).get("extra", {}).get("headers")
    if headers:
        extra["headers"] = dict(headers)
    if effort:
        extra["body"] = {"reasoning_effort": effort}
    key = str(body.get("api_key") or "").strip()
    if existing is None and not key:
        raise HTTPException(status_code=400, detail="enter the API key")
    if existing is not None and not key and existing.get("api_key_set") \
            and base_url != str(existing.get("base_url") or "").rstrip("/"):
        # A saved key only ever goes to the URL it was saved with.
        raise HTTPException(status_code=400, detail="type the API key again when you change the URL")
    pid = pid or _new_id(name)
    fields: dict[str, Any] = {"id": pid, "name": name, "kind": "openai_compat", "model_id": model,
                              "base_url": base_url, "extra": extra}
    if key:
        fields["api_key"] = key
    elif body.get("clear_key"):
        if pid == assistant.assistant_provider_id():
            raise HTTPException(status_code=409, detail="this connection is the main assistant: pick another main "
                                                        "assistant before removing its key")
        fields["api_key"] = ""
    mgr.upsert_provider(**fields)
    assistant.drop_client(pid)                       # the next call builds a client with the new values
    if (mgr.active() or {}).get("id") == pid:
        mgr.unload()
    return pid


@router.post("/api/settings/connections")
async def create_connection(request: Request):
    body = await _json(request)

    def _apply() -> dict[str, Any]:
        return {"id": _upsert_connection(None, body), **_state()}

    return await asyncio.to_thread(_apply)


@router.put("/api/settings/connections/{pid}")
async def update_connection(pid: str, request: Request):
    body = await _json(request)

    def _apply() -> dict[str, Any]:
        return {"id": _upsert_connection(pid, body), **_state()}

    return await asyncio.to_thread(_apply)


@router.delete("/api/settings/connections/{pid}")
async def delete_connection(pid: str):
    """Remove an API connection and its key. Jobs that used it go back to the main assistant (pictures: off)."""
    def _apply() -> dict[str, Any]:
        mgr = get_manager()
        row = mgr.get_provider(pid)
        if row is None:
            raise HTTPException(status_code=404, detail=f"unknown connection {pid!r}")
        if row.get("kind") != "openai_compat":
            raise HTTPException(status_code=400, detail="the local model cannot be removed (it can be changed)")
        if pid == assistant.assistant_provider_id():
            raise HTTPException(status_code=409, detail="this connection is the main assistant: pick another main "
                                                        "assistant first")
        if (mgr.active() or {}).get("id") == pid:
            mgr.unload()
        assistant.forget_connection(pid)
        assistant.drop_client(pid)
        mgr.delete_provider(pid)
        return _state()

    return await asyncio.to_thread(_apply)


def _probe_config(body: dict[str, Any]) -> ProviderConfig:
    """Config to test: typed-in values over the stored connection ``body["id"]`` (its stored key when none is typed)."""
    mgr = get_manager()
    pid = str(body.get("id") or "").strip()
    row = (mgr.get_provider(pid) or {}) if pid else {}
    extra_row = row.get("extra") or {}
    preset = next((p for p in PROVIDER_PRESETS if p["id"] == (body.get("preset") or extra_row.get("preset"))), None)
    extra: dict[str, Any] = {"retries": 1, "timeout": 60}
    headers = (preset or {}).get("extra", {}).get("headers") or extra_row.get("headers")
    if headers:
        extra["headers"] = dict(headers)
    effort = str(body.get("reasoning_effort", (extra_row.get("body") or {}).get("reasoning_effort")) or "").strip()
    if effort:
        extra["body"] = {"reasoning_effort": effort.lower()}
    base_url = _validate_url(str(body.get("base_url") or row.get("base_url") or "").strip())
    key = str(body.get("api_key") or "").strip()
    if not key and row:
        # A saved key only ever goes to the URL it was saved with: testing another URL needs the key typed again.
        if base_url != str(row.get("base_url") or "").rstrip("/"):
            raise HTTPException(status_code=400, detail="the saved key is only sent to its own URL: type the key "
                                                        "again to test a different URL")
        key = mgr._api_key_for(pid)
    return ProviderConfig(
        id=pid or "connection-test", name="connection test", kind="openai_compat",
        model_id=str(body.get("model") or row.get("model_id") or "").strip(),
        base_url=base_url, api_key=key, extra=extra,
    )


@router.post("/api/settings/connections/test")
async def test_connection(request: Request):
    """One real chat call with the typed-in (or stored) values: proves URL, key, model and latency."""
    cfg = _probe_config(await _json(request))
    if not cfg.model_id or not cfg.api_key:
        return JSONResponse({"ok": False, "detail": "enter the model name and API key first"}, status_code=400)
    provider = OpenAICompatProvider(cfg)

    def _ping() -> dict[str, Any]:
        t0 = time.perf_counter()
        try:
            reply = provider.chat([{"role": "user", "content": "Reply with exactly: pong"}], max_tokens=512, temperature=0)
        except ProviderError as exc:
            return {"ok": False, "detail": str(exc), "code": "provider_failed",
                    "latency_ms": int((time.perf_counter() - t0) * 1000)}
        finally:
            provider.unload()
        return {"ok": True, "reply": reply[:80], "latency_ms": int((time.perf_counter() - t0) * 1000)}

    result = await asyncio.to_thread(_ping)
    return JSONResponse(result, status_code=200 if result["ok"] else 502)


@router.post("/api/settings/connections/models")
async def list_connection_models(request: Request):
    """Model ids the provider offers (``GET {base}/models``), so the user picks instead of typing."""
    cfg = _probe_config(await _json(request))
    if not cfg.api_key:
        return JSONResponse({"ok": False, "detail": "enter the API key first"}, status_code=400)
    headers = {"Authorization": f"Bearer {cfg.api_key}"}

    def _fetch() -> dict[str, Any]:
        try:
            r = requests.get(cfg.base_url + "/models", headers=headers, timeout=20)
        except requests.RequestException as exc:
            return {"ok": False, "detail": f"{type(exc).__name__} calling {cfg.base_url}/models",
                    "code": "provider_unreachable"}
        if r.status_code >= 400:
            return {"ok": False, "detail": f"the provider answered HTTP {r.status_code}: {' '.join(r.text.split())[:200]}",
                    "code": "provider_failed"}
        try:
            data = r.json().get("data") or []
            return {"ok": True, "models": sorted(str(m["id"]) for m in data if isinstance(m, dict) and m.get("id"))}
        except (ValueError, AttributeError):
            return {"ok": False, "detail": "the provider's /models answer was not the OpenAI list shape",
                    "code": "provider_failed"}

    result = await asyncio.to_thread(_fetch)
    return JSONResponse(result, status_code=200 if result["ok"] else 502)
