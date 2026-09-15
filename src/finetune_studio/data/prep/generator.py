"""Resolve a chat callable for data-prep Q&A generation.

The Inference page loads into the global ``inference_engine``
(``finetune_studio.webui.app``). The Chat tab / provider UI can load into
``models.manager.get_manager()``. DataPrepRunner, ``/data-prep/start``, and
agent chat (when ``provider_id`` is omitted) must accept either — a user who
only loaded on Inference still has a usable generator.

This module never loads a model. Callers that need an explicit provider must
pass ``provider_id`` and load themselves.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any, TypedDict

# chat(messages, max_tokens, temperature, top_p) -> str
ChatFn = Callable[..., str]

NO_MODEL_MSG = (
    "No model loaded — load a model on the Inference page "
    "(or pick one in the Chat tab) before starting prep."
)


class LoadedBackend(TypedDict, total=False):
    """Already-resident chat backend (no load performed)."""

    kind: str  # "provider" | "global"
    manager: Any
    provider_id: str | None
    engine: Any


def _manager_backend() -> LoadedBackend | None:
    from finetune_studio.models.manager import get_manager

    mgr = get_manager()
    active = mgr.active()
    if active is None:
        return None
    return {
        "kind": "provider",
        "manager": mgr,
        "provider_id": active.get("id") or "",
    }


def _inference_backend() -> LoadedBackend | None:
    # Lazy import: avoid importing webui.app at module load (cycle with routes).
    from finetune_studio.webui.app import inference_engine

    if getattr(inference_engine, "model", None) is None:
        return None
    return {
        "kind": "global",
        "engine": inference_engine,
        "provider_id": None,
    }


def resolve_loaded_backend(*, prefer_inference: bool = False) -> LoadedBackend | None:
    """Return a backend for an already-loaded model, or None. Never loads.

    Default preference (data-prep start / runner):
      1. manager when ``active()`` is set
      2. ``inference_engine`` when ``.model`` is loaded

    ``prefer_inference=True`` (agent chat with omitted ``provider_id``):
      1. ``inference_engine`` when loaded — the model the user put on Inference
      2. manager when ``active()`` is set

    Preferring Inference when both are resident avoids agent chat silently
    answering with a leftover manager GGUF (and never auto-loads a second model).
    """
    if prefer_inference:
        return _inference_backend() or _manager_backend()
    return _manager_backend() or _inference_backend()


def resolve_generator() -> ChatFn | None:
    """Return a chat callable from manager or inference_engine, else None.

    Preference order matches :func:`resolve_loaded_backend` (manager first).
    """
    backend = resolve_loaded_backend()
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
