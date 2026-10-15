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
    from finetune_studio.webui.routes import hf_models

    model = _stage(monkeypatch, tmp_path)
    calls: list[int] = []
    monkeypatch.setattr(hf_models, "_refresh_model_registry", lambda: calls.append(1) or 0)
    r = client.delete("/api/hf/local/Org/Tiny")
    assert r.status_code == 200 and r.json()["deleted"] == "Org/Tiny"
    assert not model.exists()
    assert calls == [1]


def test_rescan_failure_does_not_fail_the_delete(client, monkeypatch, tmp_path) -> None:
    from finetune_studio.webui.routes import hf_models

    model = _stage(monkeypatch, tmp_path)

    def boom() -> int:
        raise OSError("disk walk failed")

    monkeypatch.setattr(hf_models, "_refresh_model_registry", boom)
    assert client.delete("/api/hf/local/Org/Tiny").status_code == 200
    assert not model.exists()


def test_missing_model_is_404_and_does_not_rescan(client, monkeypatch, tmp_path) -> None:
    from finetune_studio.webui.routes import hf_models

    monkeypatch.setattr(hf_models, "_LOCAL", tmp_path)
    calls: list[int] = []
    monkeypatch.setattr(hf_models, "_refresh_model_registry", lambda: calls.append(1) or 0)
    assert client.delete("/api/hf/local/Org/Nope").status_code == 404
    assert calls == []
