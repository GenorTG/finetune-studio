"""Resolve a chat callable for data-prep Q&A generation.

Data-prep and LLM-assisted suite generation require the configured local
27B GGUF helper (``models.helper``). This module never loads a model and
never silently falls back to a different loaded model (e.g. a project's
merged 4B on the Inference tab).

Callers that need a non-helper backend must pass an explicit ``provider_id``
(or ``external_api``) themselves.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any, TypedDict

from finetune_studio.models.helper import (
    DEFAULT_HELPER_LABEL,
    DEFAULT_HELPER_PROVIDER_ID,
    get_configured_helper_provider,
    is_helper_gguf_path,
    is_helper_provider,
    no_helper_message,
    wrong_model_message,
)

# chat(messages, max_tokens, temperature, top_p) -> str
ChatFn = Callable[..., str]

# Public alias — data-prep errors name the helper explicitly.
NO_MODEL_MSG = no_helper_message()
HELPER_NO_MODEL_MSG = NO_MODEL_MSG
WRONG_MODEL_MSG_PREFIX = "Data-prep / suite generation requires the configured helper"


class LoadedBackend(TypedDict, total=False):
    """Already-resident chat backend (no load performed)."""

    kind: str  # "provider" | "global"
    manager: Any
    provider_id: str | None
    engine: Any
    helper_label: str


def _manager_helper_backend() -> LoadedBackend | None:
    from finetune_studio.models.manager import get_manager

    mgr = get_manager()
    active = mgr.active()
    if active is None or not is_helper_provider(active):
        return None
    return {
        "kind": "provider",
        "manager": mgr,
        "provider_id": active.get("id") or DEFAULT_HELPER_PROVIDER_ID,
        "helper_label": DEFAULT_HELPER_LABEL,
    }


def _inference_helper_backend() -> LoadedBackend | None:
    # Lazy import: avoid importing webui.app at module load (cycle with routes).
    from finetune_studio.webui.app import inference_engine

    if getattr(inference_engine, "model", None) is None:
        return None
    path = getattr(inference_engine, "model_path", None) or ""
    if not is_helper_gguf_path(path):
        return None
    return {
        "kind": "global",
        "engine": inference_engine,
        "provider_id": DEFAULT_HELPER_PROVIDER_ID,
        "helper_label": DEFAULT_HELPER_LABEL,
    }


def _loaded_non_helper_path() -> str | None:
    """Path of a loaded non-helper model, if any (for clear error messages)."""
    from finetune_studio.models.manager import get_manager

    mgr = get_manager()
    active = mgr.active()
    if active is not None and not is_helper_provider(active):
        return str(
            active.get("model_id")
            or active.get("model_path")
            or active.get("id")
            or ""
        )

    from finetune_studio.webui.app import inference_engine

    if getattr(inference_engine, "model", None) is not None:
        path = getattr(inference_engine, "model_path", None) or ""
        if path and not is_helper_gguf_path(path):
            return str(path)
    return None


def resolve_helper_backend() -> LoadedBackend | None:
    """Return a backend only when the configured helper is loaded.

    Preference:
      1. ModelManager active provider when it is the helper
      2. ``inference_engine`` when its path is the helper GGUF

    Never returns a non-helper loaded model (no silent fallback).
    """
    return _manager_helper_backend() or _inference_helper_backend()


def resolve_loaded_backend(*, prefer_inference: bool = False) -> LoadedBackend | None:
    """Compatibility wrapper — always resolves the helper only.

    ``prefer_inference`` is accepted for call-site compatibility but ignored:
    helper resolution never silently prefers a non-helper Inference load.
    """
    _ = prefer_inference
    return resolve_helper_backend()


def helper_resolution_error() -> str:
    """Explain why helper resolution failed (wrong model vs not loaded)."""
    other = _loaded_non_helper_path()
    if other:
        return wrong_model_message(other)
    cfg = get_configured_helper_provider()
    if cfg is None:
        return (
            f"{HELPER_NO_MODEL_MSG} "
            f"(no provider row for '{DEFAULT_HELPER_PROVIDER_ID}' yet)."
        )
    return HELPER_NO_MODEL_MSG


def resolve_generator() -> ChatFn | None:
    """Return a chat callable for the loaded helper, else None."""
    backend = resolve_helper_backend()
    if backend is None:
        return None

    if backend["kind"] == "provider":
        mgr = backend["manager"]

        def _mgr_chat(
            messages: list[dict[str, Any]],
            max_tokens: int = 1200,
            temperature: float = 0.7,
            top_p: float = 0.9,
            **_: Any,
        ) -> str:
            return mgr.chat(
                messages,
                max_tokens=max_tokens,
                temperature=temperature,
                top_p=top_p,
            )

        return _mgr_chat

    engine = backend["engine"]

    def _engine_chat(
        messages: list[dict[str, Any]],
        max_tokens: int = 1200,
        temperature: float = 0.7,
        top_p: float = 0.9,
        **_: Any,
    ) -> str:
        return engine.generate(
            messages,
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
        )

    return _engine_chat
