"""E2E-40: the uvicorn / TestClient process must never import unsloth."""

from __future__ import annotations

import importlib
import sys
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest


def test_app_and_inference_engine_never_import_unsloth(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Importing the WebUI + constructing InferenceEngine must leave unsloth out."""
    # Drop any prior import from other tests in this session.
    for key in list(sys.modules):
        if key == "unsloth" or key.startswith("unsloth."):
            del sys.modules[key]

    # Stub heavy optional deps the app may touch at import/lifespan.
    monkeypatch.setitem(
        sys.modules,
        "transformers",
        SimpleNamespace(
            AutoModelForCausalLM=MagicMock(),
            AutoTokenizer=MagicMock(),
            BitsAndBytesConfig=MagicMock(),
        ),
    )

    # ModelManager caches a process-wide singleton (models/manager.py's
    # module-level `_manager`), and its `.engine` property lazily caches
    # the InferenceEngine instance it builds the first time it's read. If
    # an earlier test in the same process constructed that singleton while
    # `InferenceEngine` was itself monkeypatched (a MagicMock subclass, as
    # several route-level tests do), the cached `.engine` stays a MagicMock
    # forever after — reverting that test's monkeypatch only un-patches the
    # class, not the already-cached instance. This test reloads webui.app
    # specifically to get a genuinely fresh import, so it must also force a
    # fresh ModelManager/engine rather than silently inheriting whatever a
    # prior test left cached.
    import finetune_studio.models.manager as mgr_mod
    monkeypatch.setattr(mgr_mod, "_manager", None)

    with patch("finetune_studio.models.registry.scan_models", return_value=[]):
        import finetune_studio.webui.app as app_module

        importlib.reload(app_module)
        from fastapi.testclient import TestClient

        with TestClient(app_module.app) as client:
            resp = client.get("/api/inference/status")
            assert resp.status_code == 200

        from finetune_studio.testing.inference import InferenceEngine

        eng = InferenceEngine()
        assert eng.model is None

    assert "unsloth" not in sys.modules
    assert not any(k.startswith("unsloth.") for k in sys.modules)
