"""Settings API — get/update app settings from the Settings page."""

from __future__ import annotations

import json
import logging
import os
from pathlib import Path
from typing import Any

from fastapi import APIRouter, HTTPException

router = APIRouter()
log = logging.getLogger(__name__)

# Settings file location (user-data preserved across updates)
SETTINGS_PATH = Path.home() / ".finetune-studio" / "settings.json"


def _load() -> dict[str, Any]:
    if not SETTINGS_PATH.exists():
        return {}
    try:
        return json.loads(SETTINGS_PATH.read_text())
    except Exception as e:
        log.warning("settings read failed: %s", e)
        return {}


def _save(data: dict[str, Any]) -> None:
    SETTINGS_PATH.parent.mkdir(parents=True, exist_ok=True)
    SETTINGS_PATH.write_text(json.dumps(data, indent=2))


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


def get_defaults() -> dict[str, Any]:
    """Return default settings."""
    return DEFAULTS.copy()


@router.get("/api/settings")
async def get_settings():
    """Return current settings merged with defaults."""
    current = _load()
    merged = {**DEFAULTS, **current}
    return merged


@router.patch("/api/settings")
async def update_settings(request: Any):
    """Partial update of settings."""
    body = await request.json()
    if not isinstance(body, dict):
        raise HTTPException(status_code=400, detail="expected JSON object")
    current = _load()
    current.update(body)
    _save(current)
    return {**DEFAULTS, **current}


@router.post("/api/settings/reload")
async def reload_settings():
    """Tell the app to reload settings from disk (e.g. after port change).

    Returns whether a restart is needed."""
    current = _load()
    needs_restart = "port" in current or "host" in current
    return {"ok": True, "needs_restart": needs_restart, "settings": {**DEFAULTS, **current}}
