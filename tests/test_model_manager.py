"""Tests for ModelManager.load() — specifically the fast-path that
skips re-loading when the same provider is already active.

The bug being tested: previously `load()` always built a fresh provider
object and called `load()` on it, which tried to instantiate a second
Llama() while the first still held the file handle / VRAM, raising
"Failed to load model from file" on every chat call after the first
probe. The fix adds an early-return fast path.
"""
from __future__ import annotations

import os
from unittest.mock import MagicMock, patch

import pytest


class FakeProvider:
    """Minimal ModelProvider-shaped stub that records load() calls."""
    instances: list = []

    def __init__(self, cfg, **kwargs):
        self.cfg = cfg
        self._loaded_at = 0.0
        self._load_calls = 0
        FakeProvider.instances.append(self)

    def is_loaded(self) -> bool:
        return self._loaded_at > 0.0

    def load(self) -> None:
        self._load_calls += 1
        self._loaded_at = 1.0  # simulate successful load

    def unload(self) -> None:
        self._loaded_at = 0.0

    def describe(self) -> dict:
        return {"id": self.cfg.id, "loaded": self.is_loaded()}

    def chat(self, messages, **gen):
        return "ok"

    def generate(self, prompt, **gen):
        return "ok"


class TestModelManagerLoadFastPath:
    """Verify load(pid) is a no-op when the same provider is already loaded."""

    def setup_method(self):
        FakeProvider.instances.clear()

    def test_second_load_same_id_does_not_reload(self, monkeypatch):
        """The core regression: two consecutive load() calls for the same
        provider ID should result in only ONE load() on the underlying
        provider object (because the first call already loaded it)."""
        from finetune_studio.models import manager as mgr_mod

        cfg_row = {"id": "local-default", "name": "Local GGUF",
                   "kind": "local_gguf",
                   "model_id": "/fake/model.gguf",
                   "base_url": "", "api_key": "", "extra": {}}

        mm = mgr_mod.ModelManager()
        mm.get_provider = MagicMock(return_value=cfg_row)
        mm.list_providers = MagicMock(return_value=[cfg_row])

        with patch.object(mgr_mod, "build_provider", FakeProvider):
            # First load — actually loads.
            mm.load("local-default")
            assert FakeProvider.instances[0]._load_calls == 1
            assert mm.active() is not None
            assert mm.active()["loaded"] is True

            # Second load for same ID — should be a no-op.
            mm.load("local-default")
            # Still only one load() call total.
            assert FakeProvider.instances[0]._load_calls == 1
            assert len(FakeProvider.instances) == 1

    def test_second_load_different_id_does_unload_then_load(self, monkeypatch):
        """When a different provider is requested, the previous one must
        be unloaded and the new one loaded."""
        from finetune_studio.models import manager as mgr_mod

        cfg_a = {"id": "provider-a", "name": "A", "kind": "local_gguf",
                 "model_id": "/fake/a.gguf",
                 "base_url": "", "api_key": "", "extra": {}}
        cfg_b = {"id": "provider-b", "name": "B", "kind": "local_gguf",
                 "model_id": "/fake/b.gguf",
                 "base_url": "", "api_key": "", "extra": {}}

        mm = mgr_mod.ModelManager()

        def fake_get(pid):
            return {"provider-a": cfg_a, "provider-b": cfg_b}.get(pid)

        mm.get_provider = MagicMock(side_effect=fake_get)
        mm.list_providers = MagicMock(return_value=[cfg_a, cfg_b])

        with patch.object(mgr_mod, "build_provider", FakeProvider):
            mm.load("provider-a")
            assert FakeProvider.instances[0]._load_calls == 1
            a_obj = FakeProvider.instances[0]

            mm.load("provider-b")
            # The first provider was unloaded (its _loaded_at reset to 0)
            assert a_obj._loaded_at == 0.0
            # A second provider object was created and loaded
            assert len(FakeProvider.instances) == 2
            assert FakeProvider.instances[1]._load_calls == 1

    def test_load_unknown_provider_raises(self):
        from finetune_studio.models import manager as mgr_mod
        mm = mgr_mod.ModelManager()
        mm.get_provider = MagicMock(return_value=None)
        with pytest.raises(ValueError, match="Unknown provider"):
            mm.load("nonexistent")
