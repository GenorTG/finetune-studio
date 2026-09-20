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
    """Per-GPU VRAM in GB. Uses torch.cuda when present, else nvidia-smi via subprocess."""
    # Path 1: torch (preferred — no extra dep, already a requirement)
    try:
        import torch  # type: ignore
        if torch.cuda.is_available():
            out = []
            for i in range(torch.cuda.device_count()):
                free_bytes, total_bytes = torch.cuda.mem_get_info(i)
                used_bytes = total_bytes - free_bytes
                total_gb = total_bytes / 1024**3
                used_gb = used_bytes / 1024**3
                name = torch.cuda.get_device_name(i)
                out.append({
                    "index": i,
                    "name": name,
                    "used_gb": round(used_gb, 2),
                    "total_gb": round(total_gb, 2),
                    "pct": round(100.0 * used_bytes / total_bytes, 1) if total_bytes else 0.0,
                })
            return out
    except Exception as e:  # noqa: BLE001
        log.debug("torch.cuda path failed, falling back to nvidia-smi: %s", e)

    # Path 2: nvidia-smi subprocess (works even without torch importable here)
    try:
        import subprocess
        r = subprocess.run(
            ["nvidia-smi", "--query-gpu=index,name,memory.used,memory.total",
             "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=2, check=True,
        )
        out = []
        for line in r.stdout.strip().splitlines():
            parts = [p.strip() for p in line.split(",")]
            if len(parts) != 4:
                continue
            idx_s, name, used_mib, total_mib = parts
            used_gb = float(used_mib) / 1024.0
            total_gb = float(total_mib) / 1024.0
            pct = (float(used_mib) / float(total_mib) * 100.0) if float(total_mib) else 0.0
            out.append({
                "index": int(idx_s),
                "name": name,
                "used_gb": round(used_gb, 2),
                "total_gb": round(total_gb, 2),
                "pct": round(pct, 1),
            })
        return out
    except Exception as e:  # noqa: BLE001
        log.debug("nvidia-smi path failed: %s", e)

    return []


@router.get("/api/system/resources")
async def resources():
    """Host RAM + per-GPU VRAM snapshot for UI bars.

    Returns:
      {
        "ram": {"available": true, "used_gb": ..., "total_gb": ..., "pct": ...},
        "vram": [{"index": 0, "name": "RTX 3090", "used_gb": ..., "total_gb": ..., "pct": ...}, ...]
      }
    """
    return {"ram": _ram(), "vram": _vram()}


@router.get("/api/system/gpu")
async def gpu_summary():
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
async def gpu_text():
    """Sub-line for dashboard stat tile — GPU name + percent (plain text)."""
    from fastapi.responses import PlainTextResponse
    vram = _vram()
    if not vram:
        return PlainTextResponse("no GPU")
    g = vram[0]
    return PlainTextResponse(f"{g['name']} · {g['pct']}%")


@router.get("/api/system/version")
async def version():
    """Exact build identity of the running service.

    ``version`` is MAJOR.MINOR.PATCH.BUILD from the repo VERSION file — the
    pre-commit hook bumps BUILD every commit, so this pins the deployed code
    to one commit. ``git_commit`` is the short SHA when git metadata exists.
    """
    from pathlib import Path

    from finetune_studio import __release_channel__, __version__

    commit = ""
    try:
        head = Path(".git") / "HEAD"
        if head.is_file():
            ref = head.read_text(encoding="utf-8").strip()
            if ref.startswith("ref:"):
                refpath = Path(".git") / ref.split(": ", 1)[1]
                if refpath.is_file():
                    commit = refpath.read_text(encoding="utf-8").strip()[:8]
    except OSError:
        pass
    return {"version": __version__, "channel": __release_channel__, "git_commit": commit}
