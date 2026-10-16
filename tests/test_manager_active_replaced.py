"""``ModelManager.active()`` after a job replaced the model in the inference worker (2026-10-16).

Judges, the Vision reader and test runs are job processes with their own proxy to the one inference worker: when
one of them loads another model, the web process's ModelManager still held its last provider and showed it as the
active provider (``loaded: false``) for the rest of the session. Now ``active()`` notices the worker holds a
different model and reports nothing active (the provider is dropped). A worker that merely unloaded (idle expiry)
keeps the provider: it reloads on demand, as before.
"""
from __future__ import annotations

import threading
from types import SimpleNamespace

from finetune_studio.models import manager as mgr_mod


class _Worker:
    def __init__(self, model_path: str | None) -> None:
        self.model = object() if model_path else None
        self.model_path = model_path


class _LocalProvider:
    _loaded_at = 0.0

    def __init__(self, model_id: str, engine: _Worker) -> None:
        self.config = SimpleNamespace(model_id=model_id, id="helper")
        self.engine = engine

    def describe(self) -> dict:
        mine = self.engine.model is not None and self.engine.model_path == self.config.model_id
        return {"kind": "local_gguf", "id": "helper", "model_id": self.config.model_id, "loaded": mine}


def _manager(provider, worker: _Worker) -> mgr_mod.ModelManager:
    m = mgr_mod.ModelManager.__new__(mgr_mod.ModelManager)
    m._lock = threading.RLock()
    m._provider = provider
    m._active_id = "helper" if provider is not None else ""
    m._engine_instance = worker
    return m


def test_a_provider_whose_model_was_replaced_in_the_worker_is_gone() -> None:
    worker = _Worker("/models/judge.gguf")
    m = _manager(_LocalProvider("/models/helper.gguf", worker), worker)
    assert m.active() is None
    assert m._provider is None and m._active_id == ""


def test_the_same_model_in_the_worker_is_the_active_provider() -> None:
    worker = _Worker("/models/helper.gguf")
    m = _manager(_LocalProvider("/models/helper.gguf", worker), worker)
    active = m.active()
    assert active and active["id"] == "helper" and active["loaded"] is True
    assert m._provider is not None


def test_an_unloaded_worker_keeps_the_provider_for_the_next_on_demand_load() -> None:
    worker = _Worker(None)
    m = _manager(_LocalProvider("/models/helper.gguf", worker), worker)
    active = m.active()
    assert active and active["id"] == "helper" and active["loaded"] is False
    assert m._provider is not None, "idle expiry is not a replacement"


def test_a_stub_engine_without_a_real_path_never_counts_as_replaced() -> None:
    """The test suite's engine is a MagicMock: its model_path is not a str, so nothing is dropped there."""
    from unittest.mock import MagicMock

    worker = MagicMock(name="inference_engine")
    m = _manager(_LocalProvider("/models/helper.gguf", worker), worker)
    active = m.active()
    assert active and active["id"] == "helper"
    assert m._provider is not None


def test_a_failing_status_probe_is_unknown_not_replaced() -> None:
    class _Broken:
        @property
        def model(self):
            raise RuntimeError("socket gone")

    worker = _Broken()
    m = _manager(_LocalProvider("/models/helper.gguf", worker), worker)
    # describe() of the real provider would also hit the engine; the fake one here reads it directly.
    worker_desc = {"kind": "local_gguf", "id": "helper", "loaded": False}
    m._provider.describe = lambda: worker_desc  # type: ignore[method-assign]
    assert m.active()["id"] == "helper" and m._provider is not None
