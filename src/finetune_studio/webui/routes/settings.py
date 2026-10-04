"""Settings API — get/update app settings from the Settings page."""

from __future__ import annotations

import json
import logging
import os
from pathlib import Path
from typing import Any

from fastapi import APIRouter, HTTPException, Request

router = APIRouter()
log = logging.getLogger(__name__)

# Settings file location (user-data preserved across updates)
SETTINGS_PATH = Path.home() / ".finetune-studio" / "settings.json"


def _load() -> dict[str, Any]:
    if not SETTINGS_PATH.exists():
        return {}
    try:
        return json.loads(SETTINGS_PATH.read_text())
    except (OSError, ValueError) as e:
        log.warning("settings read failed: %s", e)
        return {}


def _save(data: dict[str, Any]) -> None:
    SETTINGS_PATH.parent.mkdir(parents=True, exist_ok=True)
    tmp = SETTINGS_PATH.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(data, indent=2))
    tmp.chmod(0o600)  # may hold the judge API key
    tmp.replace(SETTINGS_PATH)


# ── Default settings (used when settings.json doesn't exist) ──
DEFAULTS: dict[str, Any] = {
    "host": "0.0.0.0",
    "port": 7860,
    "cors_origins": [],
    "cors_allow_credentials": True,
    "trusted_hosts": [],
    "proxy_headers": False,
    "root_path": "",
}


# ── AI judge configuration ──
# Stored in settings.json under judge_*; env vars (FTS_JUDGE_*) are the
# defaults. The API key is write-only: it is never returned by any endpoint.
JUDGE_MODES = ("none", "heuristic", "ai", "local")
_JUDGE_KEY_FIELD = "judge_api_key"


def get_judge_config() -> dict[str, Any]:
    """Effective judge config: saved settings over env-var defaults.

    Includes the real ``api_key`` — server-side use only, never serialize it.
    """
    from finetune_studio.testing import judge as _j

    saved = _load()
    key = str(saved.get(_JUDGE_KEY_FIELD) or "").strip() or os.environ.get("FTS_JUDGE_API_KEY", "")
    mode = str(saved.get("judge_mode") or "").strip().lower()
    if mode not in JUDGE_MODES:
        mode = "ai" if key else "heuristic"
    return {
        "mode": mode,
        "model": str(saved.get("judge_model") or "").strip()
        or os.environ.get("FTS_JUDGE_MODEL", _j.DEFAULT_JUDGE_MODEL),
        "api_url": (
            str(saved.get("judge_api_url") or "").strip()
            or os.environ.get("FTS_JUDGE_API", _j.DEFAULT_JUDGE_API)
        ).rstrip("/"),
        "api_key": key,
    }


def public_judge_config() -> dict[str, Any]:
    """Judge config safe to send to the browser (key reduced to set/not set)."""
    cfg = get_judge_config()
    key_set = bool(cfg.pop("api_key"))
    cfg["api_key_set"] = key_set
    return cfg


def _redacted(current: dict[str, Any]) -> dict[str, Any]:
    out = {**DEFAULTS, **current}
    out.pop(_JUDGE_KEY_FIELD, None)
    out["judge"] = public_judge_config()
    return out


@router.get("/api/settings")
async def get_settings():
    """Return current settings merged with defaults (secrets redacted)."""
    return _redacted(_load())


@router.get("/api/settings/judge")
async def get_judge_settings():
    """Judge config for the Settings page; key shown only as set/not set."""
    return public_judge_config()


@router.patch("/api/settings")
async def update_settings(request: Request):
    """Partial update of settings."""
    try:
        body = await request.json()
    except ValueError:
        raise HTTPException(status_code=400, detail="invalid JSON") from None
    if not isinstance(body, dict):
        raise HTTPException(status_code=400, detail="expected JSON object")
    clear_key = bool(body.pop("judge_api_key_clear", False))
    mode = body.get("judge_mode")
    if mode is not None and str(mode).strip().lower() not in JUDGE_MODES:
        raise HTTPException(status_code=400, detail=f"judge_mode must be one of {JUDGE_MODES}")
    if _JUDGE_KEY_FIELD in body and not str(body[_JUDGE_KEY_FIELD] or "").strip():
        body.pop(_JUDGE_KEY_FIELD)  # blank = keep the stored key
    current = _load()
    current.update(body)
    if clear_key:
        current.pop(_JUDGE_KEY_FIELD, None)
    _save(current)
    return _redacted(current)


@router.post("/api/settings/reload")
async def reload_settings():
    """Tell the app to reload settings from disk (e.g. after port change).

    Returns whether a restart is needed."""
    current = _load()
    needs_restart = "port" in current or "host" in current
    return {"ok": True, "needs_restart": needs_restart, "settings": _redacted(current)}
