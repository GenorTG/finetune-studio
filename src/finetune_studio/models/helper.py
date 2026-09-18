"""Configured local helper model (8B GGUF) for data-prep / suite generation.

The studio keeps one dedicated provider row (``local-default``) pointing at the
helper GGUF used to mine Q&A and (when LLM-assisted) build test suites.
Callers must resolve that helper explicitly — never silently reuse whatever
happens to be loaded on the Inference tab.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

# Stable provider id seeded by ModelManager._ensure_db.
DEFAULT_HELPER_PROVIDER_ID: str = "local-default"

# Clear UI / error-message label (not the bare filename alone).
DEFAULT_HELPER_LABEL: str = "Helper · Qwen3-8B GGUF"

# Helper seat is the 8B Q5_K_M fetched from Qwen/Qwen3-8B-GGUF (2026-09-18).
# Genor's fleet rule: helpers must be <27B; the 27B stays off the helper seat.
DEFAULT_HELPER_GGUF_BASENAME: str = "Qwen3-8B-Q5_K_M.gguf"

# Loader defaults for the seeded local_gguf provider.
# 32k ctx per Genor (agentic tool-calling needs room); q8_0 KV cache keeps
# the f16 KV cache from eating ~9.4GB at that depth (8 = GGML q8_0).
DEFAULT_HELPER_EXTRA: dict[str, Any] = {
    "n_ctx": 32768,
    "n_gpu_layers": 99,
    "n_batch": 512,
    "n_threads": 0,
    "seed": -1,
    "rope_freq_base": 0.0,
    "rope_freq_scale": 0.0,
    "flash_attn": True,
    "mmap": True,
    "mlock": False,
    "type_k": 8,
    "type_v": 8,
}


def default_helper_gguf_path() -> str:
    """Absolute path to the configured helper GGUF on this host."""
    override = (os.environ.get("FTS_HELPER_GGUF") or "").strip()
    if override:
        return str(Path(override).expanduser())
    return str(
        Path.home()
        / "finetune-studio"
        / "models"
        / "gguf"
        / DEFAULT_HELPER_GGUF_BASENAME
    )


def normalize_model_path(path: str | None) -> str:
    """Absolute normalised path for equality checks (empty if unset)."""
    raw = (path or "").strip()
    if not raw:
        return ""
    p = Path(raw).expanduser()
    if not p.is_absolute():
        p = Path.cwd() / p
    return os.path.normpath(str(p))


def paths_match(a: str | None, b: str | None) -> bool:
    """True when two model paths refer to the same file/dir."""
    na = normalize_model_path(a)
    nb = normalize_model_path(b)
    if not na or not nb:
        return False
    return na == nb


def helper_basename(path: str | None) -> str:
    """Last path segment of a model path (GGUF filename or dir name)."""
    raw = (path or "").strip().rstrip("/\\")
    if not raw:
        return ""
    return Path(raw).name


def is_helper_gguf_path(path: str | None) -> bool:
    """True when ``path`` is the configured helper GGUF (env or default)."""
    if paths_match(path, default_helper_gguf_path()):
        return True
    # Basename match covers relative vs absolute layout differences.
    return helper_basename(path) == DEFAULT_HELPER_GGUF_BASENAME


def is_helper_provider(row: dict[str, Any] | None) -> bool:
    """True when a provider row is the configured local helper."""
    if not row:
        return False
    if (row.get("id") or "") == DEFAULT_HELPER_PROVIDER_ID:
        return True
    return is_helper_gguf_path(row.get("model_id") or row.get("model_path"))


def helper_display_label(
    *,
    name: str | None = None,
    model_id: str | None = None,
) -> str:
    """Human-readable helper label for UI / API (always prefixed Helper ·)."""
    cleaned = (name or "").strip()
    if cleaned.startswith("Helper"):
        return cleaned
    if (
        cleaned
        and cleaned not in ("Local GGUF", "local", DEFAULT_HELPER_PROVIDER_ID)
        and ("27B" in cleaned or "helper" in cleaned.lower() or "8B" in cleaned)
    ):
        return f"Helper · {cleaned}"
    base = helper_basename(model_id) or DEFAULT_HELPER_GGUF_BASENAME
    stem = base[: -len(".gguf")] if base.lower().endswith(".gguf") else base
    return f"Helper · {stem}"


def annotate_provider(row: dict[str, Any]) -> dict[str, Any]:
    """Copy a provider dict and add ``is_helper`` + display ``label``."""
    out = dict(row)
    helper = is_helper_provider(out)
    out["is_helper"] = helper
    if helper:
        # Keep DB name in sync with the clear label for UI pickers.
        if (out.get("name") or "").strip() in ("", "Local GGUF", "local"):
            out["name"] = DEFAULT_HELPER_LABEL
        out["label"] = helper_display_label(
            name=out.get("name"),
            model_id=out.get("model_id") or out.get("model_path"),
        )
    else:
        out["label"] = out.get("name") or out.get("id") or ""
    return out


def get_configured_helper_provider() -> dict[str, Any] | None:
    """Return the helper provider row from ModelManager, or None."""
    from finetune_studio.models.manager import get_manager

    mgr = get_manager()
    by_id = mgr.get_provider(DEFAULT_HELPER_PROVIDER_ID)
    if by_id is not None:
        return annotate_provider(by_id)
    for row in mgr.list_providers():
        if is_helper_provider(row):
            return annotate_provider(row)
    return None


def wrong_model_message(loaded_path: str | None = None) -> str:
    """Error when a non-helper model is loaded for a helper-only workflow."""
    loaded = helper_basename(loaded_path) or (loaded_path or "another model")
    return (
        f"Data-prep / suite generation requires the configured helper "
        f"({DEFAULT_HELPER_LABEL}), not {loaded}. "
        f"Load provider '{DEFAULT_HELPER_PROVIDER_ID}' "
        f"(or the matching GGUF on Inference) and retry."
    )


def no_helper_message() -> str:
    """Error when the helper is not loaded."""
    return (
        f"No helper model loaded — load {DEFAULT_HELPER_LABEL} "
        f"(provider '{DEFAULT_HELPER_PROVIDER_ID}') on Inference "
        f"or via /api/providers/{DEFAULT_HELPER_PROVIDER_ID}/load "
        f"before starting prep or LLM-assisted suite generation."
    )
