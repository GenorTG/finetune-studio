"""``/api/system/compute-device`` — choose which compute device the app uses at its next start."""

from __future__ import annotations

import logging
from typing import Any

from fastapi import APIRouter, HTTPException

from finetune_studio.accel import saved_choice
from finetune_studio.webui import compute_device as cd

router = APIRouter()
log = logging.getLogger(__name__)


@router.get("/api/system/compute-device")
async def get_compute_device() -> dict[str, Any]:
    """Detected GPUs, the saved choice, what this process runs, and whether a restart is needed.

    ``saved`` is applied only at process start; ``overridden_by_env`` is true when an explicit
    variable (FTS_GPU_DEVICES, FTS_GPU_EXCLUDE, FTS_DEVICE, CUDA_VISIBLE_DEVICES, …) beats it.
    """
    return cd.report(saved_choice.load())


@router.put("/api/system/compute-device")
async def put_compute_device(body: cd.ComputeDeviceIn) -> dict[str, Any]:
    """Persist the choice (``auto`` | ``gpu`` + ``device_id`` | ``cpu``); 400 with a reason if invalid."""
    devices = cd.list_devices()
    try:
        choice = cd.build_choice(body, devices)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from None
    try:
        saved_choice.save(choice)
    except OSError as exc:
        raise HTTPException(status_code=500, detail=f"could not save the choice: {exc}") from None
    log.info("compute device choice saved: %s", choice)
    return cd.report(choice, devices)
