"""Resolve a chat callable for data-prep Q&A generation.

Data-prep and LLM-assisted suite generation run on the job's connection
(``models.assistant``: the main assistant unless the job names its own). An
API connection is called through its own client; a local GGUF must already be
loaded. This module never loads a local model and never silently falls back to
a different loaded model (e.g. a project's merged 4B on the Inference tab).

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
    get_helper_provider_id,
    is_helper_gguf_path,
    is_helper_provider,
    no_helper_message,
    wrong_model_message,
)

# chat(messages, max_tokens, temperature, top_p) -> str
ChatFn = Callable[..., str]

# Public alias — data-prep errors name the helper explicitly.
NO_MODEL_MSG = no_helper_message(DEFAULT_HELPER_PROVIDER_ID, DEFAULT_HELPER_LABEL)
HELPER_NO_MODEL_MSG = NO_MODEL_MSG
WRONG_MODEL_MSG_PREFIX = "Data-prep / suite generation requires the configured helper"


class LoadedBackend(TypedDict, total=False):
    """Already-resident chat backend (no load performed)."""

    kind: str  # "provider" | "global"
    manager: Any
    provider_id: str | None
    engine: Any
    helper_label: str


def _manager_helper_backend(job: str | None = None) -> LoadedBackend | None:
    from finetune_studio.models import assistant
    from finetune_studio.models.manager import get_manager

    mgr = get_manager()
    pid = get_helper_provider_id(job)
    seat = mgr.get_provider(pid) or {}
    if seat.get("kind") == "openai_compat":
        # An API connection has its own client: no load, no VRAM, and the model on the card stays where it is.
        client = assistant.connection_client(pid)
        return {"kind": "provider", "manager": client, "provider_id": pid,
                "helper_label": assistant.connection_label(seat)}
    active = mgr.active()
    # The local GGUF is never loaded implicitly, and a model someone else loaded is never displaced.
    if active is None or active.get("kind") == "openai_compat" or not is_helper_provider(active):
        return None
    return {
        "kind": "provider",
        "manager": mgr,
        "provider_id": active.get("id") or pid,
        "helper_label": active.get("label") or active.get("name") or DEFAULT_HELPER_LABEL,
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


def resolve_helper_backend(job: str | None = None) -> LoadedBackend | None:
    """Return a backend for ``job`` (default: the main assistant) only when it is usable now.

    Preference:
      1. the job's API connection (its own client)
      2. ModelManager active provider when it is the local helper
      3. ``inference_engine`` when its path is the helper GGUF

    Never returns a non-helper loaded model (no silent fallback).
    """
    backend = _manager_helper_backend(job)
    if backend is not None:
        return backend
    seat = get_configured_helper_provider(job) or {}
    return None if seat.get("kind") == "openai_compat" else _inference_helper_backend()


def helper_resolution_error(job: str | None = None) -> str:
    """Explain why helper resolution failed (wrong model vs not loaded)."""
    other = _loaded_non_helper_path()
    if other:
        return wrong_model_message(other)
    cfg = get_configured_helper_provider(job)
    if cfg is None:
        return (
            f"{no_helper_message(get_helper_provider_id(job))} "
            f"(no provider row for '{get_helper_provider_id(job)}' yet)."
        )
    return no_helper_message(cfg.get("id"), cfg.get("label"))


def resolve_generator(job: str | None = None) -> ChatFn | None:
    """Return a chat callable for ``job``'s connection (default: the main assistant), else None.

    The job's extra instructions (Settings -> AI assistant) go in front of every call."""
    from finetune_studio.models.assistant import instruct

    inner = _resolve_raw_generator(job)
    if inner is None or not job:
        return inner

    def _with_instructions(messages: list[dict[str, Any]], *args: Any, **kw: Any) -> str:
        return inner(instruct(job, messages), *args, **kw)

    return _with_instructions


def _resolve_raw_generator(job: str | None) -> ChatFn | None:
    backend = resolve_helper_backend(job)
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
