"""Compute-device report + validation behind ``/api/system/compute-device`` (no route logic here).

Single responsibility: list the GPUs that physically exist (also those this process has masked,
so they can be chosen), compare the saved choice with what this process started with, and turn a
request into a ``SavedChoice``. Detection uses vendor CLIs only (``nvidia-smi`` / ``rocm-smi``): no
CUDA context is created for any card. The choice itself is applied at the next process start
(``accel.env.apply_device_policy``); nothing here changes the running process.
"""
from __future__ import annotations

import logging
import os
from typing import Any, Literal

from pydantic import BaseModel

from finetune_studio.accel.env import AppliedPolicy, get_applied
from finetune_studio.accel.saved_choice import SavedChoice, device_id
from finetune_studio.webui import gpu_probe

log = logging.getLogger(__name__)

RESTART_COMMAND = "fts restart web"  # same under systemd and a manual start; applies the Settings choice
UNSUPERVISED_RESTART = "stop the server and start it again (it is not running under a supervisor)"


def restart_command() -> str:
    """The restart that applies a saved choice: the supervisor's, or a plain instruction when unsupervised."""
    from finetune_studio.supervisor.paths import socket_path

    return RESTART_COMMAND if socket_path().exists() else UNSUPERVISED_RESTART
_VISIBILITY_VARS = ("CUDA_VISIBLE_DEVICES", "HIP_VISIBLE_DEVICES")


class ComputeDeviceIn(BaseModel):
    """PUT body: ``gpu`` needs ``device_id`` (an ``id`` from GET), ``auto``/``cpu`` must not have one."""
    mode: Literal["auto", "gpu", "cpu"]
    device_id: str | None = None


class DeviceInfo(BaseModel):
    id: str
    index: int
    name: str
    vendor: str
    runtime: str
    total_gb: float
    free_gb: float
    visible: bool      # this process can see the card (not masked by a visibility variable)
    active: bool       # the card this process computes on


# ── detection (no CUDA context) ────────────────────────────────────────────
def _nvidia_rows() -> list[dict[str, Any]]:
    txt = gpu_probe._run(["nvidia-smi", "--query-gpu=index,uuid,name,memory.total,memory.free",
                          "--format=csv,noheader,nounits"], timeout=3)
    rows: list[dict[str, Any]] = []
    for line in (txt or "").strip().splitlines():
        p = [x.strip() for x in line.split(",", 4)]
        if len(p) == 5 and p[0].isdigit():
            try:
                rows.append({"index": int(p[0]), "ident": p[1], "name": p[2], "vendor": "nvidia",
                             "total_gb": round(float(p[3]) / 1024, 2), "free_gb": round(float(p[4]) / 1024, 2)})
            except ValueError:
                continue
    return rows


def _amd_rows() -> list[dict[str, Any]]:
    return [{"index": e["index"], "ident": str(e["index"]), "name": e["name"], "vendor": "amd",
             "total_gb": e["total_gb"], "free_gb": round(e["total_gb"] - e["used_gb"], 2)}
            for e in gpu_probe._rocm_smi()]


def _visible(row: dict[str, Any], environ: dict[str, str] | os._Environ[str]) -> bool:
    var = "CUDA_VISIBLE_DEVICES" if row["vendor"] == "nvidia" else "HIP_VISIBLE_DEVICES"
    if var not in environ:
        return True
    tokens = [t.strip() for t in environ[var].split(",") if t.strip()]
    return any(t == str(row["index"]) if t.isdigit() else row["ident"].lower().startswith(t.lower())
               for t in tokens)


def list_devices() -> list[DeviceInfo]:
    """Every NVIDIA/AMD GPU on the host, ``active`` marking the one the accelerator runs on."""
    from finetune_studio.accel import get_accelerator
    acc = get_accelerator()
    rows = _nvidia_rows() + _amd_rows()
    for r in rows:
        r["visible"] = _visible(r, os.environ)
    mine = [r for r in rows if r["visible"] and r["name"] == acc.name] if acc.is_gpu else []
    active = None
    if mine:
        active = mine[0] if len(mine) == 1 else mine[min(acc.index, len(mine) - 1)]
    return [DeviceInfo(
        id=device_id(r["vendor"], r["ident"]), index=r["index"], name=r["name"],
        vendor=r["vendor"].upper() if r["vendor"] == "amd" else "NVIDIA",
        runtime=acc.runtime if acc.kind in ("cuda", "rocm") and r is active
        else ("CUDA" if r["vendor"] == "nvidia" else "ROCm"),
        total_gb=r["total_gb"], free_gb=r["free_gb"], visible=r["visible"], active=r is active,
    ) for r in rows]


# ── validation ─────────────────────────────────────────────────────────────
def build_choice(body: ComputeDeviceIn, devices: list[DeviceInfo]) -> SavedChoice:
    """Turn a request into a ``SavedChoice``; ``ValueError`` carries the 400 reason."""
    if body.mode == "gpu":
        if not body.device_id:
            raise ValueError("mode 'gpu' needs a device_id (an id from GET /api/system/compute-device)")
        if not devices:
            raise ValueError("no selectable GPU was detected on this host (nvidia-smi / rocm-smi)")
        dev = next((d for d in devices if d.id == body.device_id), None)
        if dev is None:
            raise ValueError(f"unknown device {body.device_id!r}; choose one of: "
                             + ", ".join(d.id for d in devices))
        vendor, _, ident = dev.id.partition(":")
        return SavedChoice(mode="gpu", vendor=vendor, uuid=ident, name=dev.name, index=dev.index)
    if body.device_id:
        raise ValueError(f"device_id is only valid with mode 'gpu', not {body.mode!r}")
    return SavedChoice(mode=body.mode)


# ── report ─────────────────────────────────────────────────────────────────
def _choice_view(c: SavedChoice, devices: list[DeviceInfo]) -> dict[str, Any]:
    return {"mode": c.mode, "device_id": c.device_id, "name": c.name,
            "available": c.mode != "gpu" or any(d.id == c.device_id for d in devices)}


def _effective(applied: AppliedPolicy, devices: list[DeviceInfo]) -> dict[str, Any]:
    from finetune_studio.accel import get_accelerator
    acc = get_accelerator()
    return {
        "source": applied.source, "kind": acc.kind, "is_gpu": acc.is_gpu, "device": acc.torch_device,
        "name": acc.name, "runtime": acc.runtime,
        "device_id": next((d.id for d in devices if d.active), ""),
        "degraded_reason": acc.degraded_reason, "note": applied.note,
        "visibility": {k: os.environ[k] for k in _VISIBILITY_VARS if k in os.environ},
    }


def report(saved: SavedChoice, devices: list[DeviceInfo] | None = None) -> dict[str, Any]:
    """GET payload: devices, saved + effective choice, ``restart_required``, ``overridden_by_env``."""
    devices = list_devices() if devices is None else devices
    applied = get_applied()
    overridden = bool(applied.env_overrides) and saved.mode != "auto"
    return {
        "devices": [d.model_dump() for d in devices],
        "saved": _choice_view(saved, devices),
        "effective": _effective(applied, devices),
        "env_overrides": list(applied.env_overrides),
        "overridden_by_env": overridden,
        # An env override is not lifted by a restart, so it never counts as "restart to apply".
        "restart_required": (not overridden) and saved != applied.saved,
        "restart_command": restart_command(),
    }
