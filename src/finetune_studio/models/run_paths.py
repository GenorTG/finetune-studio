"""Which folder of a training run holds the model a loader can open."""
from __future__ import annotations

import os


def is_model_dir(path: str) -> bool:
    """True when ``path`` itself is something a loader opens: a full model or a LoRA adapter."""
    return os.path.isfile(os.path.join(path, "config.json")) or os.path.isfile(
        os.path.join(path, "adapter_config.json")
    )


def has_model_files(path: str) -> bool:
    """False only for an existing folder with nothing a loader could open; a file or a repo id is the loader's call."""
    if not os.path.isdir(path) or is_model_dir(path):
        return True
    try:
        return any(name.endswith((".gguf", ".safetensors")) for name in os.listdir(path))
    except OSError:
        return True


def resolve_run_model_dir(path: str) -> str:
    """``path`` if it is already a model; for a run's output folder its ``merged/``, then its ``adapter/``.

    A run's output folder holds those two next to checkpoints and exports but no model of its own, so opening it
    directly fails with an opaque tokenizer or "Unrecognized model" error. Anything else comes back unchanged.
    """
    if not os.path.isdir(path) or is_model_dir(path):
        return path
    merged = os.path.join(path, "merged")
    if os.path.isfile(os.path.join(merged, "config.json")):
        return merged
    adapter = os.path.join(path, "adapter")
    if os.path.isfile(os.path.join(adapter, "adapter_config.json")):
        return adapter
    return path
