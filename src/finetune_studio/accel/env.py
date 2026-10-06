"""GPU visibility policy — which physical GPUs this process may touch.

Single responsibility: translate ``FTS_GPU_DEVICES`` (allowlist) and
``FTS_GPU_EXCLUDE`` (denylist) into the vendor visibility env vars
(``CUDA_VISIBLE_DEVICES`` / ``HIP_VISIBLE_DEVICES``) *before* torch or
llama.cpp initialise a driver. Masking at the env level is the only way to
keep llama.cpp (which splits layers across every device it sees) and torch
off a GPU that is reserved for something else.

Tokens match case-insensitively against the GPU name (substring), the index,
or the UUID prefix, e.g. ``FTS_GPU_EXCLUDE="GTX 1070"`` or ``FTS_GPU_DEVICES=0,1``.
A number that is the index of a listed GPU matches that index only ("0" must not
hit "RTX 30**9**0"); any other number ("1070") is a name fragment.
A policy that would hide every GPU is ignored (CPU is only for GPU-less hosts);
an explicit ``CUDA_VISIBLE_DEVICES`` / ``HIP_VISIBLE_DEVICES`` always wins.
"""
from __future__ import annotations

import json
import logging
import os
import shutil
import subprocess
from collections.abc import Callable, Mapping, MutableMapping
from dataclasses import dataclass

from finetune_studio.accel.device import Accelerator

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


def matches(gpu: PhysicalGPU, tokens: list[str], indices: frozenset[int]) -> bool:
    """True when any policy token selects ``gpu`` (``indices`` = every listed GPU index)."""
    uuid = gpu.ident.lower()
    for t in tokens:
        if t.isdigit() and int(t) in indices:
            if int(t) == gpu.index:
                return True
            continue  # an index token never falls through to name/UUID matching
        if t in gpu.name.lower() or (len(t) >= 4 and uuid.startswith(t)):
            return True
    return False


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
    indices = frozenset(g.index for g in gpus)
    kept = [g for g in gpus if not allow or matches(g, allow, indices)]
    return [g for g in kept if not (deny and matches(g, deny, indices))]


def isolated_env(acc: Accelerator, environ: Mapping[str, str] | None = None) -> dict[str, str]:
    """Env vars that make ``acc`` the *only* device (index 0) a child process can see.

    Libraries hard-code device 0 (transformers' ``Trainer``/``TrainingArguments``, accelerate,
    bitsandbytes, unsloth), so a child that must train on another card is started with that card
    remapped to index 0 — the one fix that covers every library. ``acc.index`` is a position in the
    *currently visible* list, so an existing ``CUDA_VISIBLE_DEVICES=1,0`` is composed, not replaced.
    ``{}`` when the device is already alone at index 0 or the backend has no such variable.
    """
    env = os.environ if environ is None else environ
    if acc.kind == "cuda":
        var = "CUDA_VISIBLE_DEVICES"
    elif acc.kind == "rocm":
        var = ("CUDA_VISIBLE_DEVICES" if "CUDA_VISIBLE_DEVICES" in env and "HIP_VISIBLE_DEVICES" not in env
               else "HIP_VISIBLE_DEVICES")
    elif acc.kind == "xpu":
        var = "ZE_AFFINITY_MASK"
    else:
        return {}
    if acc.index == 0 and acc.device_count <= 1:
        return {}
    listed = [t.strip() for t in env.get(var, "").split(",") if t.strip()]
    return {var: listed[acc.index] if acc.index < len(listed) else str(acc.index)}


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
        if var in env or (var == "HIP_VISIBLE_DEVICES" and "ROCR_VISIBLE_DEVICES" in env):
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
