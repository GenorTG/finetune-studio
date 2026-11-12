"""Settings API — get/update app settings from the Settings page."""

from __future__ import annotations

import json
import logging
from typing import Any

from fastapi import APIRouter, HTTPException, Request

from finetune_studio.access import paths as access_paths

router = APIRouter()
log = logging.getLogger(__name__)

# Settings file location (user-data preserved across updates)
SETTINGS_PATH = access_paths.root() / "settings.json"     # $FTS_ROOT, like every other piece of app state


# Secrets an older version kept in settings.json as plain text; removed from the file the first time it is read.
_LEGACY_SECRET_KEYS = ("judge_api_key",)


def _load() -> dict[str, Any]:
    if not SETTINGS_PATH.exists():
        return {}
    try:
        data = json.loads(SETTINGS_PATH.read_text())
    except (OSError, ValueError) as e:
        log.warning("settings read failed: %s", e)
        return {}
    if isinstance(data, dict) and any(k in data for k in _LEGACY_SECRET_KEYS):
        data = {k: v for k, v in data.items() if k not in _LEGACY_SECRET_KEYS}
        _save(data)
        log.info("removed a plain-text legacy API key from %s", SETTINGS_PATH)
    return data


def _save(data: dict[str, Any]) -> None:
    SETTINGS_PATH.parent.mkdir(parents=True, exist_ok=True)
    tmp = SETTINGS_PATH.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(data, indent=2))
    tmp.chmod(0o600)
    tmp.replace(SETTINGS_PATH)


# ── Default settings (used when settings.json doesn't exist) ──
# The listen address is NOT here: it lives in $FTS_ROOT/server.json (access.server, Settings -> Access & network).
DEFAULTS: dict[str, Any] = {
    "cors_origins": [],
    "cors_allow_credentials": True,
    "proxy_headers": False,
}


# ── Test judging ──
# Judging is a separate step after a test run. The judge is any provider row (default: the helper seat);
# auto-judge makes a finished run start that step by itself. Off by default: a run ends at the raw transcripts.
_TEST_AUTO_JUDGE = "test_auto_judge"
_TEST_JUDGE_PROVIDER = "test_judge_provider_id"


def get_test_settings() -> dict[str, Any]:
    """Effective test-judging settings. ``judge_provider_id`` '' means "use the main assistant".

    The judge's connection is the ``judge`` job of the assistant (``models.assistant``); a choice saved here by an
    older version moves there the first time it is read."""
    from finetune_studio.models import assistant

    saved = _load()
    legacy = str(saved.get(_TEST_JUDGE_PROVIDER) or "").strip()
    if _TEST_JUDGE_PROVIDER in saved:
        try:
            if legacy and not assistant.job_setting("judge"):
                assistant.set_job("judge", provider_id=legacy)
        except ValueError:
            log.warning("dropped the saved judge %r: the connection no longer exists", legacy)
        _save({k: v for k, v in saved.items() if k != _TEST_JUDGE_PROVIDER})
    return {
        "auto_judge": bool(saved.get(_TEST_AUTO_JUDGE, False)),
        "judge_provider_id": assistant.job_setting("judge"),
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
        from finetune_studio.models import assistant

        try:
            assistant.set_job("judge", provider_id=str(body["judge_provider_id"] or "").strip())
        except ValueError as e:
            raise HTTPException(status_code=400, detail=str(e)) from None
    _save(current)
    return await get_testing_settings()


# How the Guide runs the actions it prepares (guide/actions.py): "ask" shows a Run / Skip card for each one (default),
# "auto" runs them at once (hands-free) and lets the Guide continue with the next step.
_GUIDE_ACTION_MODE = "guide_action_mode"
GUIDE_ACTION_MODES = ("ask", "auto")


def get_guide_action_mode() -> str:
    mode = str(_load().get(_GUIDE_ACTION_MODE) or "ask")
    return mode if mode in GUIDE_ACTION_MODES else "ask"


@router.get("/api/settings/guide")
async def get_guide_settings():
    """The Guide's action mode: ``ask`` (confirm each action) or ``auto`` (hands-free)."""
    return {"action_mode": get_guide_action_mode(), "modes": list(GUIDE_ACTION_MODES)}


@router.put("/api/settings/guide")
async def put_guide_settings(request: Request):
    try:
        body = await request.json()
    except ValueError:
        raise HTTPException(status_code=400, detail="invalid JSON") from None
    if not isinstance(body, dict):
        raise HTTPException(status_code=400, detail="expected JSON object")
    mode = body.get("action_mode")
    if mode not in GUIDE_ACTION_MODES:
        raise HTTPException(status_code=400, detail=f"action_mode must be one of {', '.join(GUIDE_ACTION_MODES)}")
    current = _load()
    current[_GUIDE_ACTION_MODE] = mode
    _save(current)
    return await get_guide_settings()


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

