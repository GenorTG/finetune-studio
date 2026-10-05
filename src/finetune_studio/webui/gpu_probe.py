"""Vendor-neutral GPU memory probes for the web UI (no route logic here).

Order of trust: torch through ``finetune_studio.accel`` (honours the policy env
and visibility vars), then vendor CLIs — ``nvidia-smi``, ``rocm-smi``,
``xpu-smi`` / ``sycl-ls`` — then Apple unified memory. A host with *any* GPU
must never come back empty merely because one vendor tool is missing.
Every probe is best-effort and never raises.
"""
from __future__ import annotations

import json
import logging
import os
import platform
import re
import shutil
import subprocess
from typing import Any

from finetune_studio.accel.env import PhysicalGPU
from finetune_studio.accel.env import matches as policy_matches

log = logging.getLogger(__name__)
GIB = 1024 ** 3
MIB = 1024 ** 2


def _run(cmd: list[str], timeout: float = 3.0) -> str | None:
    if shutil.which(cmd[0]) is None:
        return None
    try:
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout, check=False)
    except (OSError, subprocess.SubprocessError):
        return None
    return r.stdout if r.returncode == 0 else None


def _entry(index: int, name: str, used_b: float, total_b: float, source: str) -> dict[str, Any]:
    return {
        "index": index, "name": name,
        "used_gb": round(used_b / GIB, 2), "total_gb": round(total_b / GIB, 2),
        "pct": round(100.0 * used_b / total_b, 1) if total_b else 0.0,
        "source": source,
    }


# ── individual probes ──────────────────────────────────────────────────────
def _torch_devices() -> list[dict[str, Any]]:
    from finetune_studio.accel import get_accelerator
    acc = get_accelerator()
    if not acc.is_gpu:
        return []
    import torch
    if acc.kind == "mps":
        import psutil
        vm = psutil.virtual_memory()
        return [_entry(0, acc.name + " (unified memory)", vm.total - vm.available, vm.total, "torch")]
    ns = getattr(torch, acc.torch_namespace or "cuda")
    out = []
    for i in range(ns.device_count()):
        free, total = ns.mem_get_info(i)
        out.append(_entry(i, ns.get_device_name(i), total - free, total, "torch"))
    return out


def _policy_allows(index: int, name: str, indices: frozenset[int]) -> bool:
    """Honour the GPU policy env on the vendor-CLI path, which sees every physical card."""
    visible = [t.strip() for t in os.environ.get("CUDA_VISIBLE_DEVICES", "").split(",") if t.strip()]
    if visible and all(t.isdigit() for t in visible) and str(index) not in visible:
        return False
    deny = [t.strip().lower() for t in os.environ.get("FTS_GPU_EXCLUDE", "").split(",") if t.strip()]
    return not (deny and policy_matches(PhysicalGPU(index, "", name), deny, indices))


def _nvidia_smi() -> list[dict[str, Any]]:
    txt = _run(["nvidia-smi", "--query-gpu=index,name,memory.used,memory.total",
                "--format=csv,noheader,nounits"], timeout=2)
    rows: list[tuple[int, str, float, float]] = []
    for line in (txt or "").strip().splitlines():
        p = [x.strip() for x in line.split(",")]
        if len(p) == 4:
            try:
                rows.append((int(p[0]), p[1], float(p[2]) * MIB, float(p[3]) * MIB))
            except ValueError:
                continue
    indices = frozenset(r[0] for r in rows)
    return [_entry(i, name, used, total, "nvidia-smi") for i, name, used, total in rows
            if _policy_allows(i, name, indices)]


def parse_rocm_smi(txt: str) -> list[dict[str, Any]]:
    """Parse ``rocm-smi --showmeminfo vram --showproductname --json``."""
    try:
        data = json.loads(txt)
    except ValueError:
        return []
    out = []
    for key, val in sorted(data.items()):
        m = re.fullmatch(r"card(\d+)", key)
        if not m or not isinstance(val, dict):
            continue
        try:
            total = float(val["VRAM Total Memory (B)"])
            used = float(val["VRAM Total Used Memory (B)"])
        except (KeyError, ValueError):
            continue
        name = val.get("Card Series") or val.get("Card series") or val.get("Card Model") or f"AMD GPU {m.group(1)}"
        out.append(_entry(int(m.group(1)), str(name), used, total, "rocm-smi"))
    return out


def _rocm_smi() -> list[dict[str, Any]]:
    txt = _run(["rocm-smi", "--showmeminfo", "vram", "--showproductname", "--json"], timeout=4)
    return parse_rocm_smi(txt) if txt else []


def parse_xpu_smi(discovery: str, stats: dict[int, str]) -> list[dict[str, Any]]:
    """``xpu-smi discovery -j`` (+ per-device ``stats -j``) -> entries (used may be 0)."""
    try:
        devs = json.loads(discovery).get("device_list", [])
    except ValueError:
        return []
    out = []
    for d in devs:
        idx = int(d.get("device_id", len(out)))
        total = float(d.get("memory_physical_size_byte") or 0)
        used = 0.0
        try:
            for m in json.loads(stats.get(idx, "{}")).get("device_level", []):
                if m.get("metrics_type") == "XPU_STATS_MEMORY_USED":
                    used = float(m.get("value", 0)) * MIB
        except ValueError:
            pass
        out.append(_entry(idx, str(d.get("device_name", f"Intel GPU {idx}")), used, total, "xpu-smi"))
    return out


def _xpu_smi() -> list[dict[str, Any]]:
    disc = _run(["xpu-smi", "discovery", "-j"], timeout=4)
    if not disc:
        return []
    try:
        ids = [int(d.get("device_id", 0)) for d in json.loads(disc).get("device_list", [])]
    except ValueError:
        return []
    stats = {i: _run(["xpu-smi", "stats", "-d", str(i), "-j"], timeout=4) or "{}" for i in ids}
    return parse_xpu_smi(disc, stats)


def parse_sycl_ls(txt: str) -> list[dict[str, Any]]:
    """Names only (sycl-ls reports no memory use): ``[level_zero:gpu][...] Intel(R) Arc(TM) ...``."""
    out = []
    for line in txt.splitlines():
        if ":gpu" in line and ("level_zero" in line or "opencl" in line):
            name = line.split("]", 2)[-1].strip()
            if name and all(name != o["name"] for o in out):
                out.append(_entry(len(out), name, 0, 0, "sycl-ls"))
    return out


def _sycl_ls() -> list[dict[str, Any]]:
    txt = _run(["sycl-ls"], timeout=5)
    return parse_sycl_ls(txt) if txt else []


def _apple() -> list[dict[str, Any]]:
    if platform.system() != "Darwin" or platform.machine() != "arm64":
        return []
    try:
        import psutil
        vm = psutil.virtual_memory()
    except Exception:  # noqa: BLE001
        return []
    return [_entry(0, "Apple GPU (unified memory)", vm.total - vm.available, vm.total, "psutil")]


# ── public API ─────────────────────────────────────────────────────────────
def vram_devices() -> list[dict[str, Any]]:
    """Per-GPU memory for every vendor; ``[]`` only when no GPU is reachable at all."""
    probes = (_torch_devices, _nvidia_smi, _rocm_smi, _xpu_smi, _sycl_ls, _apple)
    for probe in probes:
        try:
            found = probe()
        except Exception as exc:  # noqa: BLE001 - one broken backend must not hide the others
            log.debug("gpu probe %s failed: %s", probe.__name__, exc)
            continue
        if found:
            return found
    return []


def free_and_consumers() -> tuple[int | None, list[dict[str, Any]]]:
    """``(free_mib, top compute consumers)`` of the active GPU, or ``(None, [])``."""
    devs = vram_devices()
    if not devs:
        return None, []
    pick = devs[0]
    try:
        from finetune_studio.accel import get_accelerator
        acc = get_accelerator()
        if acc.is_gpu:
            pick = next((d for d in devs if d["index"] == acc.index), devs[0])
    except Exception:  # noqa: BLE001, S110
        pass
    free_mib = int((pick["total_gb"] - pick["used_gb"]) * 1024)
    return free_mib, _consumers()[:5]


def _consumers() -> list[dict[str, Any]]:
    top: list[dict[str, Any]] = []
    txt = _run(["nvidia-smi", "--query-compute-apps=pid,used_memory", "--format=csv,noheader,nounits"], 5)
    for line in (txt or "").splitlines():
        try:
            pid_s, mem_s = line.split(",")
            top.append({"pid": int(pid_s.strip()), "vram_mib": int(mem_s.strip())})
        except ValueError:
            continue
    if not top:
        top = parse_rocm_pids(_run(["rocm-smi", "--showpids"], 5) or "")
    return sorted(top, key=lambda t: -t["vram_mib"])


def parse_rocm_pids(txt: str) -> list[dict[str, Any]]:
    """``rocm-smi --showpids`` rows: ``PID  NAME  GPUs  VRAM USED(B) ...``."""
    out = []
    for line in txt.splitlines():
        m = re.match(r"^\s*(\d+)\s+(\S+)\s+\d+\s+(\d+)\b", line)
        if m:
            out.append({"pid": int(m.group(1)), "vram_mib": int(int(m.group(3)) / MIB)})
    return out


def debug_gpus() -> list[dict[str, Any]]:
    """Shape used by the debug-info page: name, vram_total_mb, vram_free_mb, driver."""
    drv = ""
    txt = _run(["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"], 3)
    if txt:
        drv = txt.splitlines()[0].strip()
    return [{
        "name": d["name"], "vram_total_mb": int(d["total_gb"] * 1024),
        "vram_free_mb": int((d["total_gb"] - d["used_gb"]) * 1024),
        "driver": drv, "source": d["source"],
    } for d in vram_devices()]
