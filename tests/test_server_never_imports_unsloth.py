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
