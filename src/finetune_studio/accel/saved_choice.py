"""Persisted compute-device choice (Auto / one specific GPU / CPU only).

Single responsibility: the tiny on-disk record behind the Settings "Compute device" card.
Stdlib only, because ``finetune_studio/__init__`` reads it (through ``accel.env``) before torch or
llama.cpp may initialise a driver. The file lives at ``<FTS_ROOT>/compute_device.json`` — a file of
its own rather than a key in ``settings.json``, whose PATCH route accepts arbitrary keys and is
anchored to ``$HOME`` instead of ``FTS_ROOT``. Applying the choice (setting the visibility env vars)
is ``accel.env``'s job; it only ever happens at process start, so a saved change needs a restart.

Explicit environment variables always win over the saved choice; ``env_overrides`` names them.
"""
from __future__ import annotations

import json
import logging
import os
from collections.abc import Mapping
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

log = logging.getLogger(__name__)

CHOICE_FILE = "compute_device.json"
MODES = ("auto", "gpu", "cpu")
# Variables that pin the device explicitly; when any is set the saved choice is not applied.
ENV_POLICY_VARS = ("FTS_GPU_DEVICES", "FTS_GPU_EXCLUDE", "FTS_DEVICE", "CUDA_VISIBLE_DEVICES",
                   "HIP_VISIBLE_DEVICES", "ROCR_VISIBLE_DEVICES")


@dataclass(frozen=True)
class SavedChoice:
    mode: str = "auto"      # auto | gpu | cpu
    vendor: str = ""        # nvidia | amd (gpu mode only)
    uuid: str = ""          # stable id: NVIDIA UUID, or the card index for AMD (PhysicalGPU.ident)
    name: str = ""
    index: int = -1         # index when saved; only a tie-breaker, indices shift with hardware

    @property
    def device_id(self) -> str:
        """Id the API exposes for the chosen GPU (``nvidia:GPU-…``); empty unless mode is gpu."""
        return device_id(self.vendor, self.uuid) if self.mode == "gpu" else ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, raw: object) -> SavedChoice:
        """Tolerant parse: anything unusable is Auto (a broken file must never block start-up)."""
        if not isinstance(raw, dict):
            return cls()
        mode = str(raw.get("mode", "auto")).strip().lower()
        if mode == "cpu":
            return cls(mode="cpu")
        if mode != "gpu":
            return cls()
        vendor, uuid = str(raw.get("vendor", "")).strip().lower(), str(raw.get("uuid", "")).strip()
        if vendor not in ("nvidia", "amd") or not uuid:
            return cls()
        idx = raw.get("index")
        return cls(mode="gpu", vendor=vendor, uuid=uuid, name=str(raw.get("name", "")).strip(),
                   index=idx if isinstance(idx, int) and not isinstance(idx, bool) else -1)


def device_id(vendor: str, ident: str) -> str:
    return f"{vendor}:{ident}"


def choice_path(environ: Mapping[str, str] | None = None) -> Path:
    env = os.environ if environ is None else environ
    root = env.get("FTS_ROOT") or str(Path.home() / ".finetune-studio")
    return Path(root) / CHOICE_FILE


def load(path: Path | None = None) -> SavedChoice:
    """The saved choice; Auto when the file is missing or unreadable."""
    p = path or choice_path()
    try:
        return SavedChoice.from_dict(json.loads(p.read_text(encoding="utf-8")))
    except FileNotFoundError:
        return SavedChoice()
    except (OSError, ValueError) as exc:
        log.warning("accel: ignoring unreadable %s (%s)", p, exc)
        return SavedChoice()


def save(choice: SavedChoice, path: Path | None = None) -> Path:
    """Atomically persist ``choice`` (tmp file + rename)."""
    p = path or choice_path()
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(choice.to_dict(), indent=2), encoding="utf-8")
    tmp.replace(p)
    return p


def env_overrides(environ: Mapping[str, str] | None = None) -> tuple[str, ...]:
    """Names of the explicit device-policy variables set in ``environ`` (they beat the saved choice)."""
    env = os.environ if environ is None else environ
    out = []
    for name in ENV_POLICY_VARS:
        if name not in env:
            continue
        if name == "FTS_DEVICE" and env[name].strip().lower() in ("", "auto"):
            continue  # the documented default value, not an override
        out.append(name)
    return tuple(out)
