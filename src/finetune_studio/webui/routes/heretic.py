"""Heretic endpoints: status, install, upstream release check.

Heretic is an *optional external tool* (AGPL-3.0) that removes a model's refusal
behaviour by searching abliteration parameters with Optuna. It is not part of
this app's own license and not installed by ``install.sh``; this module is the
whole surface that reads, installs and updates it.

Every handler is a plain ``def``: :func:`heretic_tool.probe` and the installer
run subprocesses (``--help``, ``git``, ``uv``), which must never run on the
event loop (``tests/test_async_handlers_no_blocking_calls.py``).
"""

from __future__ import annotations

import logging
from typing import Any

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from finetune_studio.training import heretic_tool

log = logging.getLogger(__name__)
router = APIRouter()


class InstallIn(BaseModel):
    """Optional pin override; the default is the commit with headless support."""

    ref: str = ""


@router.get("/api/system/heretic")
def heretic_status() -> dict[str, Any]:
    """Is the tool installed, can it run unattended, and what does GitHub have?"""
    status = heretic_tool.probe()
    return status.as_dict()


@router.get("/api/system/heretic/upstream")
def heretic_upstream() -> dict[str, Any]:
    """Heretic's newest release and newest commit on GitHub (best-effort)."""
    status = heretic_tool.probe()
    upstream = heretic_tool.check_upstream()
    return {
        **upstream.as_dict(),
        "installed_version": status.version,
        "installed_ref": status.ref,
        "update_available": heretic_tool.update_available(status, upstream),
        "pinned": heretic_tool.PINNED_LABEL,
    }


@router.post("/api/system/heretic/install")
def heretic_install(body: InstallIn | None = None) -> dict[str, Any]:
    """Install or re-install the tool into ``$FTS_ROOT/heretic``.

    Synchronous on purpose: it is a few hundred MB over the network (about a
    minute here) and the Export page waits for the answer either way. A failure
    returns 500 with Heretic's own reason, never a 200 that hides it.
    """
    ref = (body.ref if body else "") or heretic_tool.PINNED_REF
    try:
        status = heretic_tool.install(ref)
    except heretic_tool.HereticError as e:
        log.warning("heretic install failed: %s", e)
        raise HTTPException(status_code=500, detail=str(e)) from e
    return {"ok": True, **status.as_dict()}


@router.delete("/api/system/heretic")
def heretic_uninstall() -> dict[str, Any]:
    """Remove the checkout and its environment (never the app's own venv)."""
    heretic_tool.uninstall()
    return {"ok": True, **heretic_tool.probe().as_dict()}
