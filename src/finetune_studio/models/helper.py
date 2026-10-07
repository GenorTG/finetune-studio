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

from finetune_studio.models.llama_loader import DEFAULT_N_CTX

# Stable provider id seeded by ModelManager._ensure_db.
DEFAULT_HELPER_PROVIDER_ID: str = "local-default"

# The helper *seat*: which provider row does the helper work (data-prep mining, suite generation,
# preference authoring, the Guide). It is the local GGUF unless the user connected an API provider
# on Settings, which fills the seat with the ``api-helper`` row. Stored in ``app_state``.
API_HELPER_PROVIDER_ID: str = "api-helper"
HELPER_SEAT_KEY: str = "helper_seat"

# Clear UI / error-message label (not the bare filename alone).
DEFAULT_HELPER_LABEL: str = "Helper · Gemma 4 12B Uncensored GGUF"

# Helper seat is the Gemma 4 12B Uncensored Q4_K_M
# (zaakirio/gemma-4-12b-it-uncensored-GGUF, fetched 2026-09-28).
# Genor's fleet rule: helpers must be <27B; the 27B stays off the helper seat.
DEFAULT_HELPER_GGUF_BASENAME: str = "gemma-4-12b-it-uncensored-Q4_K_M.gguf"

# Alternate helper — bigger MoE, available as a second local provider so the
# user can swap via the Inference picker without re-downloading.
# Qwen3-30B-A3B IQ4_XS (unsloth dynamic, ~16 GB on disk) — fits 16 GB VRAM with
# a small KV cache; the 30B weights MUST all be memory-resident (3B active is
# compute, not memory).
ALTERNATE_HELPER_PROVIDER_ID: str = "local-qwen30b-a3b"
ALTERNATE_HELPER_LABEL: str = "Helper · Qwen3-30B-A3B Uncensored GGUF"
ALTERNATE_HELPER_GGUF_BASENAME: str = "Qwen3-30B-A3B-Instruct-2507-IQ4_XS.gguf"

# Loader defaults for the seeded local_gguf provider.
# 32k ctx per Genor (agentic tool-calling needs room); q8_0 KV cache keeps
# the f16 KV cache from eating ~9.4GB at that depth (8 = GGML q8_0).
DEFAULT_HELPER_EXTRA: dict[str, Any] = {
    "n_ctx": DEFAULT_N_CTX,
    "n_gpu_layers": -1,  # all layers — real count comes from the GGUF header
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

# Alternate helper (Qwen3-30B-A3B) gets the same loader defaults. The IQ4_XS
# quant is small enough that f16 KV cache fits at 16k ctx on 16 GB; use q8_0
# for the same reason as the default — ~9 GB f16 KV at 32k would evict
# weights from a 16 GB card.
ALTERNATE_HELPER_EXTRA: dict[str, Any] = dict(DEFAULT_HELPER_EXTRA)


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


def alternate_helper_gguf_path() -> str:
    """Absolute path to the secondary helper GGUF on this host."""
    override = (os.environ.get("FTS_ALT_HELPER_GGUF") or "").strip()
    if override:
        return str(Path(override).expanduser())
    return str(
        Path.home()
        / "finetune-studio"
        / "models"
        / "gguf"
        / ALTERNATE_HELPER_GGUF_BASENAME
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
    if paths_match(path, alternate_helper_gguf_path()):
        return True
    # Basename match covers relative vs absolute layout differences.
    base = helper_basename(path)
    return base in (DEFAULT_HELPER_GGUF_BASENAME, ALTERNATE_HELPER_GGUF_BASENAME)


def get_helper_provider_id() -> str:
    """Id of the provider row that currently holds the helper seat (default: the local GGUF)."""
    from finetune_studio.models.manager import ModelManager, get_manager

    seat = ModelManager.get_state(HELPER_SEAT_KEY, DEFAULT_HELPER_PROVIDER_ID) or DEFAULT_HELPER_PROVIDER_ID
    if seat != DEFAULT_HELPER_PROVIDER_ID and get_manager().get_provider(seat) is None:
        return DEFAULT_HELPER_PROVIDER_ID  # seat row deleted: never leave the app without a helper
    return seat


def set_helper_provider_id(pid: str) -> None:
    """Seat ``pid`` as the helper; it must be an existing provider row."""
    from finetune_studio.models.manager import ModelManager, get_manager

    if get_manager().get_provider(pid) is None:
        raise ValueError(f"Unknown provider: {pid}")
    ModelManager.set_state(HELPER_SEAT_KEY, pid)


def is_helper_provider(row: dict[str, Any] | None) -> bool:
    """True when a provider row is a configured helper (local default/alternate, or the seated API provider)."""
    if not row:
        return False
    pid = (row.get("id") or "")
    if pid in (DEFAULT_HELPER_PROVIDER_ID, ALTERNATE_HELPER_PROVIDER_ID):
        return True
    if pid == API_HELPER_PROVIDER_ID:
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
        and (
            "27B" in cleaned
            or "helper" in cleaned.lower()
            or any(tag in cleaned for tag in ("8B", "12B", "A3B"))
        )
    ):
        return f"Helper · {cleaned}"
    # When given only a path/model_id, map known helper basenames to the
    # canonical label so the UI doesn't show lowercase "12b" / "a3b" fragments
    # in the basename.
    base = helper_basename(model_id)
    if base == DEFAULT_HELPER_GGUF_BASENAME:
        return DEFAULT_HELPER_LABEL
    if base == ALTERNATE_HELPER_GGUF_BASENAME:
        return ALTERNATE_HELPER_LABEL
    stem = (base or DEFAULT_HELPER_GGUF_BASENAME)[: -len(".gguf")] if (
        base or ""
    ).lower().endswith(".gguf") else (base or DEFAULT_HELPER_GGUF_BASENAME)
    return f"Helper · {stem}"


def annotate_provider(row: dict[str, Any]) -> dict[str, Any]:
    """Copy a provider dict and add ``is_helper`` + display ``label``."""
    out = dict(row)
    helper = is_helper_provider(out)
    out["is_helper"] = helper
    if out.get("id") == API_HELPER_PROVIDER_ID:
        out["label"] = f"Helper · API · {out.get('model_id') or 'model not set'}"
    elif helper:
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
    by_id = mgr.get_provider(get_helper_provider_id())
    if by_id is not None:
        return annotate_provider(by_id)
    for row in mgr.list_providers():
        if is_helper_provider(row):
            return annotate_provider(row)
    return None


def _seat_label() -> str:
    row = get_configured_helper_provider() or {}
    return row.get("label") or DEFAULT_HELPER_LABEL


def wrong_model_message(loaded_path: str | None = None) -> str:
    """Error when a non-helper model is loaded for a helper-only workflow."""
    loaded = helper_basename(loaded_path) or (loaded_path or "another model")
    return (
        f"Data-prep / suite generation requires the configured helper "
        f"({_seat_label()}), not {loaded}. "
        f"Load provider '{get_helper_provider_id()}' "
        f"(or the matching GGUF on Inference) and retry."
    )


def no_helper_message(pid: str | None = None, label: str | None = None) -> str:
    """Error when the helper is not loaded (names the seated provider unless both are given)."""
    pid = pid or get_helper_provider_id()
    return (
        f"No helper model loaded — load {label or _seat_label()} "
        f"(provider '{pid}') on Inference "
        f"or via /api/providers/{pid}/load "
        f"before starting prep or LLM-assisted suite generation."
    )


def missing_gguf_for_provider(pid: str) -> str:
    """Path of a ``local_gguf`` provider's model file when it does not exist, else ''."""
    from finetune_studio.models.manager import get_manager

    row = get_manager().get_provider(pid)
    if not row or row.get("kind") != "local_gguf":
        return ""
    path = str(row.get("model_id") or "").strip()
    return path if path and not Path(path).expanduser().is_file() else ""


def helper_missing_message(path: str) -> str:
    """Actionable text for a helper GGUF that is not on disk."""
    return (
        "No helper model is installed yet. Q&A generation needs a chat-capable GGUF "
        f"(expected {path}). Download one in Model library (a 7B-12B instruct GGUF "
        "works), then choose it as the helper on this page, or set FTS_HELPER_GGUF."
    )
