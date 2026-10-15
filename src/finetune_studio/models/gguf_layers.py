"""Read real model topology from GGUF headers.

Why: loader extra historically used magic n_gpu_layers=99 for "all layers",
which lies in the UI and hides the model's real depth. gguf (pip gguf) ships
with llama-cpp-python, so we read the native value:

- key is prefixed by architecture, e.g. ``qwen3.block_count`` (Qwen family)
  or ``llama.block_count`` (LLaMA family) — never assume one prefix.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from gguf import GGUFReader

_GGUF_SUFFIXES = (".gguf",)


def is_gguf_path(model_path: str) -> bool:
    return Path(model_path).suffix.lower() in _GGUF_SUFFIXES


def gguf_header_values(path: str) -> dict[str, Any]:
    """Return selected integer header values for ``path`` (empty on any error).

    Reading the header is mmap-light (no tensor data touched).
    """
    out: dict[str, Any] = {}
    try:
        reader = GGUFReader(path)
    except Exception:  # noqa: BLE001 -unreadable/corrupt GGUF → caller falls back
        return out

    for field in reader.fields.values():
        # Each field's parts: [name_len, name_bytes, type, value]. The last
        # part holds the typed value. Scalar uint32/uint64 → int.
        parts = field.parts
        if not parts:
            continue
        value = parts[-1]
        if getattr(value, "shape", None) == (1,) and value.dtype.itemsize in (4, 8):
            try:
                out[field.name] = int(value[0])
            except Exception:  # noqa: BLE001, S112 -skip a malformed field, keep scanning
                continue
    return out


def resolve_block_count(model_path: str) -> dict[str, Any]:
    """Return ``{"block_count": int|None, "context_length": int|None}`` for a GGUF.

    Never raises — missing/unreadable file yields empty values so the loader
    can fall back to the caller's configured n_gpu_layers unchanged.
    """
    if not is_gguf_path(model_path):
        return {"block_count": None, "context_length": None}
    values = gguf_header_values(model_path)
    block_count = None
    context_length = None
    for key, val in values.items():
        name = key.rsplit(".", 1)[-1]
        if name == "block_count" and block_count is None:
            block_count = val
        elif name == "context_length" and context_length is None:
            context_length = val
    return {"block_count": block_count, "context_length": context_length}
