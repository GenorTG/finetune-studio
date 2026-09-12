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


@router.get("/api/settings")
async def get_settings():
    """Return current settings."""
    return _load()


@router.patch("/api/settings")
async def update_settings(request: Any):
    """Partial update of settings."""
    body = await request.json()
    if not isinstance(body, dict):
        raise HTTPException(status_code=400, detail="expected JSON object")
    current = _load()
    current.update(body)
    _save(current)
    return current
