"""Settings API — get/update app settings from the Settings page."""

from __future__ import annotations

import json
import logging
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
    tmp.chmod(0o600)
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


# ── Test judging ──
# Judging is a separate step after a test run. The judge is any provider row (default: the helper seat);
# auto-judge makes a finished run start that step by itself. Off by default: a run ends at the raw transcripts.
_TEST_AUTO_JUDGE = "test_auto_judge"
_TEST_JUDGE_PROVIDER = "test_judge_provider_id"


def get_test_settings() -> dict[str, Any]:
    """Effective test-judging settings. ``judge_provider_id`` '' means "use the helper seat"."""
    saved = _load()
    return {
        "auto_judge": bool(saved.get(_TEST_AUTO_JUDGE, False)),
        "judge_provider_id": str(saved.get(_TEST_JUDGE_PROVIDER) or "").strip(),
    }


# Keys the pre-run-then-judge settings.json may still carry; never echoed (one held an API key).
_LEGACY_JUDGE_KEYS = ("judge_api_key", "judge_mode", "judge_provider", "judge_model", "judge_base_url")


def _redacted(current: dict[str, Any]) -> dict[str, Any]:
    out = {**DEFAULTS, **current}
    for key in _LEGACY_JUDGE_KEYS:
        out.pop(key, None)
    out["testing"] = get_test_settings()
    return out


@router.get("/api/settings")
async def get_settings():
    """Return current settings merged with defaults (secrets redacted)."""
    return _redacted(_load())


@router.get("/api/settings/testing")
async def get_testing_settings():
    """Test-judging settings plus the provider rows that can judge (helper seat first)."""
    from finetune_studio.testing.judge import (
        default_judge_provider_id,
        list_judge_providers,
    )

    return {**get_test_settings(), "effective_judge_provider_id": default_judge_provider_id(),
            "providers": list_judge_providers()}


@router.put("/api/settings/testing")
async def put_testing_settings(request: Request):
    """Save ``auto_judge`` and/or ``judge_provider_id`` ('' = helper seat)."""
    try:
        body = await request.json()
    except ValueError:
        raise HTTPException(status_code=400, detail="invalid JSON") from None
    if not isinstance(body, dict):
        raise HTTPException(status_code=400, detail="expected JSON object")
    current = _load()
    if "auto_judge" in body:
        if not isinstance(body["auto_judge"], bool):
            raise HTTPException(status_code=400, detail="auto_judge must be true or false")
        current[_TEST_AUTO_JUDGE] = body["auto_judge"]
    if "judge_provider_id" in body:
        pid = str(body["judge_provider_id"] or "").strip()
        if pid:
            from finetune_studio.models.manager import get_manager

            if get_manager().get_provider(pid) is None:
                raise HTTPException(status_code=400, detail=f"unknown provider '{pid}'")
        current[_TEST_JUDGE_PROVIDER] = pid
    _save(current)
    return await get_testing_settings()


@router.patch("/api/settings")
async def update_settings(request: Request):
    """Partial update of settings."""
    try:
        body = await request.json()
    except ValueError:
        raise HTTPException(status_code=400, detail="invalid JSON") from None
    if not isinstance(body, dict):
        raise HTTPException(status_code=400, detail="expected JSON object")
    current = _load()
    current.update(body)
    _save(current)
    return _redacted(current)


@router.post("/api/settings/reload")
async def reload_settings():
    """Tell the app to reload settings from disk (e.g. after port change).

    Returns whether a restart is needed."""
    current = _load()
    needs_restart = "port" in current or "host" in current
    return {"ok": True, "needs_restart": needs_restart, "settings": _redacted(current)}
