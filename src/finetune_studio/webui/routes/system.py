"""System resources endpoint — host RAM + per-GPU VRAM snapshot.

Lightweight: called every ~3s by page templates to refresh bars.
"""

from __future__ import annotations

import logging

from fastapi import APIRouter

router = APIRouter()
log = logging.getLogger(__name__)


def _ram() -> dict:
    """Host RAM in GB."""
    try:
        import psutil  # type: ignore
    except Exception:  # noqa: BLE001
        log.warning("psutil not installed — RAM stats unavailable")
        return {"available": False, "used_gb": 0.0, "total_gb": 0.0, "pct": 0.0}
    vm = psutil.virtual_memory()
    total_gb = vm.total / 1024**3
    used_gb = (vm.total - vm.available) / 1024**3
    return {
        "available": True,
        "used_gb": round(used_gb, 2),
        "total_gb": round(total_gb, 2),
        "pct": round(vm.percent, 1),
    }


def _vram() -> list[dict]:
    """Per-GPU VRAM in GB for any vendor (torch/accel, nvidia-smi, rocm-smi, xpu-smi, Apple)."""
    from finetune_studio.webui.gpu_probe import vram_devices
    return vram_devices()


def _accel_summary() -> dict:
    """Compact accelerator state for the Host Resources strip (cached detection)."""
    try:
        from finetune_studio.accel import get_accelerator
        a = get_accelerator()
        return {"kind": a.kind, "name": a.name, "runtime": a.runtime, "is_gpu": a.is_gpu,
                "device": a.torch_device, "degraded_reason": a.degraded_reason}
    except Exception as exc:  # noqa: BLE001 - the strip must render even if accel breaks
        log.warning("accelerator summary failed: %s", exc)
        return {"kind": "unknown", "name": "", "runtime": "", "is_gpu": False, "device": "",
                "degraded_reason": ""}


# GPU probes (torch / CUDA init on first use, nvidia-smi) are blocking: these handlers are plain ``def`` so FastAPI runs
# them in its threadpool. As ``async def`` the first probe after a restart, while the worker loaded a 12B, held the event
# loop 33 s; the supervisor's health check timed out and killed the web server twice (2026-10-19).
@router.get("/api/system/resources")
def resources():
    """Host RAM + per-GPU VRAM snapshot for UI bars.

    Returns:
      {
        "ram": {"available": true, "used_gb": ..., "total_gb": ..., "pct": ...},
        "vram": [{"index": 0, "name": "RTX 3090", "used_gb": ..., "total_gb": ..., "pct": ...}, ...],
        "accelerator": {"kind": "cuda|rocm|xpu|mps|cpu", "device": "cuda:0", "degraded_reason": ""}
      }
    """
    return {"ram": _ram(), "vram": _vram(), "accelerator": _accel_summary()}


@router.get("/api/system/accelerator")
def accelerator():
    """Full accelerator report: chosen device, torch build, llama.cpp backends, hardware,
    policy env and ``degraded_reason`` (non-empty when a GPU host is running on CPU)."""
    from finetune_studio import accel
    d = accel.describe()
    d["degraded_reason"] = d["accelerator"].get("degraded_reason", "")
    return d


@router.get("/api/system/gpu")
def gpu_summary():
    """Short VRAM string for dashboard stat tile (plain text for SPA poll).

    Returns e.g. "12.3 / 23.5 GB" or "—" when no GPU.
    """
    from fastapi.responses import PlainTextResponse
    vram = _vram()
    if not vram:
        return PlainTextResponse("—")
    g = vram[0]
    return PlainTextResponse(f"{g['used_gb']:.1f} / {g['total_gb']:.1f} GB")


@router.get("/api/system/gpu-text")
def gpu_text():
    """Sub-line for dashboard stat tile — GPU name + percent (plain text)."""
    from fastapi.responses import PlainTextResponse
    vram = _vram()
    if not vram:
        return PlainTextResponse("no GPU")
    g = vram[0]
    return PlainTextResponse(f"{g['name']} · {g['pct']}%")


@router.get("/api/health")
async def health():
    """Liveness for the supervisor probe: answers from the event loop, touches no GPU, DB or model."""
    return {"status": "ok"}


@router.get("/api/system/version")
async def version():
    """Exact build identity of the running service.

    ``version`` is the build string (``1.2.3`` on a tagged release, ``1.2.3+<branch>.<sha8>`` in a
    dev checkout), ``release`` the SemVer from VERSION, ``git_commit`` the short SHA. All of it is
    re-read per request (from files, no subprocess), so a long-lived service never reports the
    build it booted with.
    """
    from finetune_studio.version import (
        build_version,
        git_info,
        is_prerelease,
        release_channel,
        release_version,
    )

    release = release_version()
    info = git_info()
    return {
        "version": build_version(),
        "release": release,
        "channel": release_channel(release),
        "prerelease": is_prerelease(release),
        "git_commit": info.commit[:8],
        "git_branch": info.branch,
    }
