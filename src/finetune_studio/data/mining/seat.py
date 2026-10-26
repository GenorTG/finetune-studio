"""Which helper a mining job talks to, written down at submit and resolved again in the mining worker process.

The web app decides at submit which helper is usable (``data.prep.generator.resolve_generator``: the seated API helper,
or the local helper GGUF already loaded in the inference worker) and stores it on each job as ``helper["target"]``:

- ``{"kind": "api", "provider_id": ...}``: the worker builds its own client from the provider row (key read server-side);
  an API helper works from any process.
- ``{"kind": "local", "model_path": ...}``: the worker sends its calls to the ONE inference worker, and only while that
  GGUF is the model loaded there. It never loads, swaps or unloads it (a local helper is never auto-loaded): when someone
  unloads or replaces it mid-mining, the calls fail with that reason and the job records it.

The Vision reader seat is written down the same way (``helper["vision"]``, ``models.vision.Seat`` as a dict).
"""
from __future__ import annotations

import logging
import threading
from collections.abc import Callable
from pathlib import Path
from typing import Any

ChatFn = Callable[..., str]
log = logging.getLogger(__name__)


def current_target() -> dict[str, Any]:
    """The helper ``resolve_generator`` would use right now, as JSON (web process, at submit)."""
    from finetune_studio.data.prep.generator import resolve_helper_backend
    from finetune_studio.models.helper import get_configured_helper_provider

    backend = resolve_helper_backend("mining")
    if backend is not None and backend.get("kind") == "provider":
        active = backend["manager"].active() or {}
        if active.get("kind") == "openai_compat":
            return {"kind": "api", "provider_id": str(active.get("id") or backend.get("provider_id") or "")}
        return {"kind": "local", "model_path": str(active.get("model_id") or active.get("model_path") or "")}
    if backend is not None:
        return {"kind": "local", "model_path": str(getattr(backend["engine"], "model_path", "") or "")}
    row = get_configured_helper_provider("mining") or {}   # the seat itself (the generator was resolved elsewhere)
    if row.get("kind") == "openai_compat":
        return {"kind": "api", "provider_id": str(row.get("id") or "")}
    return {"kind": "local", "model_path": str(row.get("model_id") or "")}


def vision_seat() -> dict[str, str]:
    """The picture reader to record on a job, with "main assistant" resolved to a concrete connection."""
    from finetune_studio.models.vision import resolve_seat

    return resolve_seat().as_dict()


_providers: dict[str, Any] = {}
_providers_lock = threading.Lock()


def _api_provider(provider_id: str) -> Any:
    from finetune_studio.models.assistant import connection_client
    from finetune_studio.models.manager import get_manager

    with _providers_lock:
        if provider_id in _providers:
            return _providers[provider_id]
        row = get_manager().get_provider(provider_id)
        if row is None or row.get("kind") != "openai_compat":
            raise RuntimeError(f"the API connection {provider_id!r} is not configured any more (Settings -> AI assistant)")
        provider = connection_client(provider_id).provider
        _providers[provider_id] = provider
        return provider


def chat_for(target: dict[str, Any]) -> ChatFn:
    """A chat callable for a recorded target (mining worker process), with the mining instructions recorded on the job
    in front of every call. Raises with the reason when it cannot exist."""
    from finetune_studio.models.assistant import instruct

    chat = _raw_chat_for(target)
    text = str(target.get("instructions") or "")
    if not text:
        return chat

    def _instructed(messages: list[dict[str, Any]], *args: Any, **kw: Any) -> str:
        return chat(instruct("mining", messages, text), *args, **kw)

    return _instructed


def _raw_chat_for(target: dict[str, Any]) -> ChatFn:
    kind = target.get("kind")
    if kind == "api":
        provider = _api_provider(str(target.get("provider_id") or ""))

        def _api_chat(messages: list[dict[str, Any]], max_tokens: int = 1200, temperature: float = 0.7,
                      top_p: float = 0.9, **_: Any) -> str:
            return provider.chat(messages, max_tokens=max_tokens, temperature=temperature, top_p=top_p)

        return _api_chat
    if kind == "local":
        from finetune_studio.inference import get_engine
        from finetune_studio.models.helper import paths_match

        want = str(target.get("model_path") or "")
        if not want:
            raise RuntimeError("the job names no local helper model")
        engine = get_engine()

        def _local_chat(messages: list[dict[str, Any]], max_tokens: int = 1200, temperature: float = 0.7,
                        top_p: float = 0.9, **_: Any) -> str:
            loaded = str(getattr(engine, "model_path", "") or "") if engine.model is not None else ""
            if not paths_match(loaded, want):
                now = f"now {Path(loaded).name}" if loaded else "nothing is loaded"
                raise RuntimeError(f"the local helper {Path(want).name} is no longer loaded in the inference worker "
                                   f"({now}); load it again and run Fill gaps")
            return engine.generate(messages, max_tokens=max_tokens, temperature=temperature, top_p=top_p,
                                   top_k=40, repeat_penalty=1.1)

        return _local_chat
    raise RuntimeError(f"unknown helper target {target!r}")


def forget_providers() -> None:
    """The worker is done: drop its API clients (a later worker reads the provider rows afresh)."""
    with _providers_lock:
        for provider in _providers.values():
            try:
                provider.unload()
            except Exception as e:  # noqa: BLE001 - closing an HTTP session is best effort
                log.debug("closing helper client failed: %s", e)
        _providers.clear()
