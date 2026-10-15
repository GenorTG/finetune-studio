"""Regression E2E-22: Inference "Unload" must also free a manager-loaded provider.

Agent chat can load a provider into the model manager; the UI has no separate
unload for it, so VRAM stayed pinned (18.9 GB) and training couldn't start.
"""
from __future__ import annotations

from unittest.mock import MagicMock

import pytest


def test_unload_frees_engine_and_manager(client, monkeypatch: pytest.MonkeyPatch) -> None:
    import finetune_studio.models.manager as manager_mod
    import finetune_studio.webui.app as app_mod

    engine = MagicMock()
    mgr = MagicMock()
    monkeypatch.setattr(app_mod, "inference_engine", engine)
    monkeypatch.setattr(manager_mod, "get_manager", lambda: mgr)

    r = client.post("/api/models/unload", json={})
    assert r.status_code == 200, r.text
    assert r.json() == {"status": "unloaded"}
    engine.unload.assert_called_once()
    mgr.unload.assert_called_once()
