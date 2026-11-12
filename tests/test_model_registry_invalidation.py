"""A CLI process must invalidate the WebUI's separate in-memory model registry."""
from __future__ import annotations


def test_webui_rescans_once_after_external_registry_invalidation(client, monkeypatch):
    from finetune_studio.models import registry as model_registry
    from finetune_studio.models.registry_invalidation import (
        invalidate_model_registry,
        read_generation,
    )
    from finetune_studio.webui.routes import models as model_routes

    scans: list[list[str]] = []
    monkeypatch.setattr(model_registry, "scan_models", lambda dirs: scans.append(list(dirs)) or [])
    generation = invalidate_model_registry()

    response = client.get("/api/system/version")
    assert response.status_code == 200
    assert scans and model_routes._registry_generation == generation
    assert read_generation() == generation

    scans.clear()
    response = client.get("/api/system/version")
    assert response.status_code == 200
    assert scans == []
