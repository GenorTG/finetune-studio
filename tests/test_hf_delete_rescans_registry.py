"""Deleting a downloaded HF model must drop it from the model registry too.

Found live: after ``DELETE /api/hf/local/<repo>`` the files were gone but /api/models/list (and every
model picker) kept offering the deleted model until a manual ``POST /api/models/refresh``.
"""
from __future__ import annotations

from pathlib import Path


def _stage(monkeypatch, tmp_path: Path) -> Path:
    from finetune_studio.webui.routes import hf_models

    monkeypatch.setattr(hf_models, "_LOCAL", tmp_path)
    model = tmp_path / "Org__Tiny"
    model.mkdir()
    (model / "config.json").write_text("{}")
    return model


def test_delete_rescans_model_registry(client, monkeypatch, tmp_path) -> None:
    from finetune_studio.inference.remote import RemoteEngine
    from finetune_studio.models import hf_cache

    model = _stage(monkeypatch, tmp_path)
    calls: list[int] = []
    monkeypatch.setattr(hf_cache, "_LOCAL", tmp_path)
    monkeypatch.setattr(RemoteEngine, "_status", lambda _self, fresh=False: {"loaded": False})
    monkeypatch.setattr("finetune_studio.models.registry_invalidation.invalidate_model_registry",
                        lambda: calls.append(1) or 1)
    r = client.delete("/api/hf/local/Org/Tiny")
    assert r.status_code == 200 and r.json()["deleted"] == "Org/Tiny"
    assert not model.exists()
    assert calls == [1]


def test_rescan_failure_does_not_fail_the_delete(client, monkeypatch, tmp_path) -> None:
    from finetune_studio.inference.remote import RemoteEngine
    from finetune_studio.models import hf_cache

    model = _stage(monkeypatch, tmp_path)
    monkeypatch.setattr(hf_cache, "_LOCAL", tmp_path)
    monkeypatch.setattr(RemoteEngine, "_status", lambda _self, fresh=False: {"loaded": False})

    def boom() -> int:
        raise OSError("invalidation failed")

    monkeypatch.setattr("finetune_studio.models.registry_invalidation.invalidate_model_registry", boom)
    assert client.delete("/api/hf/local/Org/Tiny").status_code == 200
    assert not model.exists()


def test_missing_model_is_404_and_does_not_rescan(client, monkeypatch, tmp_path) -> None:
    from finetune_studio.models import hf_cache

    monkeypatch.setattr(hf_cache, "_LOCAL", tmp_path)
    calls: list[int] = []
    monkeypatch.setattr("finetune_studio.models.registry_invalidation.invalidate_model_registry",
                        lambda: calls.append(1) or 1)
    assert client.delete("/api/hf/local/Org/Nope").status_code == 404
    assert calls == []


def test_invalid_repo_id_is_400_and_does_not_touch_cache(client, monkeypatch, tmp_path) -> None:
    from finetune_studio.webui.routes import hf_models

    monkeypatch.setattr(hf_models, "_LOCAL", tmp_path)
    response = client.delete("/api/hf/local/Org/Bad%20Repo")
    assert response.status_code == 400
    assert "invalid repo_id" in response.json()["detail"]
    assert list(tmp_path.iterdir()) == []


def test_loaded_or_loading_cache_model_cannot_be_deleted(client, monkeypatch, tmp_path) -> None:
    from finetune_studio.inference.remote import RemoteEngine
    from finetune_studio.models import hf_cache

    model = _stage(monkeypatch, tmp_path)
    monkeypatch.setattr(hf_cache, "_LOCAL", tmp_path)
    monkeypatch.setattr(RemoteEngine, "_status", lambda _self, fresh=False: {
        "loaded": True, "model_path": str(model / "model.safetensors"),
    })
    response = client.delete("/api/hf/local/Org/Tiny")
    assert response.status_code == 409
    assert "currently loaded" in response.json()["detail"]
    assert model.exists()


def test_local_list_resolves_hub_ids_only_when_unambiguous(monkeypatch, tmp_path) -> None:
    from finetune_studio import db
    from finetune_studio.webui.routes import hf_models

    monkeypatch.setattr(hf_models, "_LOCAL", tmp_path)
    for name in ("Org__Tiny", "Org__Ambiguous", "untracked-cache"):
        (tmp_path / name).mkdir()
    monkeypatch.setattr(db, "list_hf_downloads_recent", lambda limit: [
        {"repo_id": "Org/Tiny"},
        {"repo_id": "Org/Ambiguous"},
        {"repo_id": "Org__Ambiguous"},
    ])

    rows = {row["cache_key"]: row for row in hf_models._local_models()}
    assert rows["Org__Tiny"]["repo_id"] == "Org__Tiny"
    assert rows["Org__Tiny"]["hub_repo_id"] == "Org/Tiny"
    assert rows["Org__Tiny"]["repo_id_known"] is True
    assert rows["Org__Ambiguous"]["repo_id_known"] is False
    assert rows["untracked-cache"]["repo_id_known"] is False
