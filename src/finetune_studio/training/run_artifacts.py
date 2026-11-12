"""Does a training run have a model to use? Read from its output folder, not from its status.

A run stopped in its first seconds, or one that failed before a checkpoint, has an output folder with nothing loadable
in it, yet the Training page offered Set production / Open in inference / Export for it (walkthrough 2026-11-12, a
15-second stopped run). ``usable`` says whether the folder holds an adapter, a checkpoint with one, a merged model or a
GGUF, and ``reason`` says why not, for the disabled buttons' titles.
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Any

_WEIGHTS = (".safetensors", ".bin")


def _has_weights(d: Path) -> bool:
    try:
        return d.is_dir() and any(f.is_file() and f.name.endswith(_WEIGHTS) for f in d.iterdir())
    except OSError:
        return False


def has_model(output_path: str | os.PathLike[str] | None) -> bool:
    if not output_path:
        return False
    root = Path(output_path)
    if not root.is_dir():
        return False
    if (root / "adapter" / "adapter_config.json").is_file() and _has_weights(root / "adapter"):
        return True
    if (root / "adapter_config.json").is_file() and _has_weights(root):
        return True
    if (root / "merged" / "config.json").is_file() and _has_weights(root / "merged"):
        return True
    if (root / "config.json").is_file() and _has_weights(root):
        return True
    try:
        if any(p.suffix == ".gguf" for p in (root / "gguf").glob("*.gguf")):
            return True
        return any((c / "adapter_config.json").is_file() and _has_weights(c) for c in root.glob("checkpoint-*"))
    except OSError:
        return False


def usable(run: dict[str, Any]) -> tuple[bool, str]:
    """(True, "") when the run has a model to load, export or promote; else (False, why not)."""
    status = str(run.get("status") or "")
    if status in ("running", "loading", "training", "saving", "queued", "pending"):
        return False, "Still running: wait until it finishes."
    if has_model(run.get("output_path")):
        return (status in ("done", "stopped")), ("" if status in ("done", "stopped")
                                                 else "It failed: its files may be incomplete. Train it again.")
    if status == "stopped":
        return False, "Stopped before a checkpoint was saved: nothing to load or export."
    if status in ("failed", "error"):
        return False, "Failed before a checkpoint was saved: nothing to load or export."
    return False, "Its output folder holds no model files."


def annotate(runs: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Each run with ``usable`` / ``usable_reason`` added (a copy; the rows are not changed)."""
    out = []
    for r in runs:
        ok, why = usable(r)
        out.append({**r, "usable": ok, "usable_reason": why})
    return out
