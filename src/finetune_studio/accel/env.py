"""GPU visibility policy — which physical GPUs this process may touch.

Single responsibility: translate ``FTS_GPU_DEVICES`` (allowlist) and
``FTS_GPU_EXCLUDE`` (denylist) into the vendor visibility env vars
(``CUDA_VISIBLE_DEVICES`` / ``HIP_VISIBLE_DEVICES``) *before* torch or
llama.cpp initialise a driver. Masking at the env level is the only way to
keep llama.cpp (which splits layers across every device it sees) and torch
off a GPU that is reserved for something else.

Tokens match case-insensitively against the GPU name (substring), the index,
or the UUID prefix, e.g. ``FTS_GPU_EXCLUDE="GTX 1070"`` or ``FTS_GPU_DEVICES=0,1``.
A policy that would hide every GPU is ignored (CPU is only for GPU-less hosts);
an explicit ``CUDA_VISIBLE_DEVICES`` / ``HIP_VISIBLE_DEVICES`` always wins.
"""
from __future__ import annotations

import json
import logging
import os
import shutil
import subprocess
from collections.abc import Callable, MutableMapping
from dataclasses import dataclass

log = logging.getLogger(__name__)

Runner = Callable[[list[str]], str]


@dataclass(frozen=True)
class PhysicalGPU:
    index: int
    ident: str   # value usable in the visibility env var (UUID for NVIDIA, index for AMD)
    name: str


def _run(cmd: list[str]) -> str:
    out = subprocess.run(cmd, capture_output=True, text=True, timeout=10, check=False)
    return out.stdout if out.returncode == 0 else ""


def _tokens(raw: str | None) -> list[str]:
    return [t.strip().lower() for t in (raw or "").split(",") if t.strip()]


def _matches(gpu: PhysicalGPU, tokens: list[str]) -> bool:
    uuid = gpu.ident.lower()
    return any(t == str(gpu.index) or t in gpu.name.lower() or (len(t) >= 4 and uuid.startswith(t))
               for t in tokens)


def list_nvidia(run: Runner = _run) -> list[PhysicalGPU]:
    if shutil.which("nvidia-smi") is None and run is _run:
        return []
    gpus: list[PhysicalGPU] = []
    for line in run(["nvidia-smi", "--query-gpu=index,uuid,name",
                     "--format=csv,noheader"]).splitlines():
        parts = [p.strip() for p in line.split(",", 2)]
        if len(parts) == 3 and parts[0].isdigit():
            gpus.append(PhysicalGPU(int(parts[0]), parts[1], parts[2]))
    return gpus


def list_amd(run: Runner = _run) -> list[PhysicalGPU]:
    if shutil.which("rocm-smi") is None and run is _run:
        return []
    try:
        data = json.loads(run(["rocm-smi", "--showproductname", "--json"]) or "{}")
    except ValueError:
        return []
    gpus: list[PhysicalGPU] = []
    for key, val in data.items():
        if key.startswith("card") and key[4:].isdigit() and isinstance(val, dict):
            name = str(val.get("Card Series") or val.get("Card series")
                       or val.get("Card Model") or key)
            gpus.append(PhysicalGPU(int(key[4:]), key[4:], name))
    return sorted(gpus, key=lambda g: g.index)


def _select(gpus: list[PhysicalGPU], allow: list[str], deny: list[str]) -> list[PhysicalGPU]:
    kept = [g for g in gpus if not allow or _matches(g, allow)]
    return [g for g in kept if not (deny and _matches(g, deny))]


def apply_device_policy(
    environ: MutableMapping[str, str] | None = None,
    nvidia: Runner = _run,
    amd: Runner = _run,
) -> dict[str, str]:
    """Apply FTS_GPU_DEVICES / FTS_GPU_EXCLUDE; return the env vars that were set."""
    env = os.environ if environ is None else environ
    allow, deny = _tokens(env.get("FTS_GPU_DEVICES")), _tokens(env.get("FTS_GPU_EXCLUDE"))
    if not allow and not deny:
        return {}
    applied: dict[str, str] = {}
    for var, lister, runner in (("CUDA_VISIBLE_DEVICES", list_nvidia, nvidia),
                                ("HIP_VISIBLE_DEVICES", list_amd, amd)):
        if var in env or (var == "CUDA_VISIBLE_DEVICES" and "ROCR_VISIBLE_DEVICES" in env):
            continue  # the operator pinned devices explicitly
        gpus = lister(runner)
        if not gpus:
            continue
        kept = _select(gpus, allow, deny)
        if not kept:
            log.warning("accel: FTS_GPU_DEVICES/FTS_GPU_EXCLUDE would hide every %s GPU; "
                        "ignoring the policy (CPU is only used on GPU-less hosts)", var.split("_")[0])
            continue
        if len(kept) == len(gpus):
            continue  # nothing masked, leave the driver default alone
        env[var] = ",".join(g.ident for g in kept)
        applied[var] = env[var]
        log.info("accel: %s=%s (masked: %s)", var, env[var],
                 ", ".join(g.name for g in gpus if g not in kept))
    return applied
