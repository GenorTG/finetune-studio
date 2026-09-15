"""E2E-21: agent chat must reuse the Inference-loaded model.

When provider_id is omitted, POST /data-prep/chat must use the already-loaded
inference_engine (even if the manager has a different inactive/active
provider registered) and must NEVER call manager.load — that was dual-loading
a 15+ GB GGUF beside the Inference model and answering with the wrong one.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any
from unittest.mock import MagicMock

import pytest

from finetune_studio.data.prep.generator import NO_MODEL_MSG


@pytest.fixture
def fts_root(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    root = tmp_path / "fts"
    root.mkdir()
    projects = root / "projects"
    projects.mkdir()
    monkeypatch.setattr("finetune_studio.data.fs.paths._ROOT", root)
    monkeypatch.setattr("finetune_studio.data.fs.paths._PROJECTS", projects)
    return root


def _project(client: Any) -> str:
    r = client.post("/api/projects", json={"name": "Agent Chat Model Res"})
    assert r.status_code in (200, 201), r.text
    return r.json()["id"]


def _loaded_engine(reply: str = "ok from inference engine") -> MagicMock:
    eng = MagicMock()
    eng.model = MagicMock()
    eng.model_path = "/models/Qwen3-4B"
    eng.model.create_chat_completion.return_value = {
        "choices": [{"message": {"content": reply}}],
    }
    return eng


def _manager_with_other_provider(*, active: dict[str, Any] | None) -> MagicMock:
    """Manager that knows about a different GGUF provider; load must not run."""
    mgr = MagicMock()
    mgr.active.return_value = active
    mgr.get_provider.return_value = {
        "id": "local-default",
        "kind": "local_gguf",
        "model_id": "/models/gguf/Qwen3.8-27B-abliterated-Q4_K_M.gguf",
        "name": "local-default",
    }
    mgr.load = MagicMock(
        side_effect=AssertionError("manager.load must not be called"),
    )
    mgr.chat.return_value = "ok from manager"
    return mgr


@pytest.mark.parametrize(
    "mgr_active",
    [
        None,  # inactive — provider registered but not loaded
        {  # active — different provider already resident in manager
            "id": "local-default",
            "kind": "local_gguf",
            "model_id": "/models/gguf/Qwen3.8-27B-abliterated-Q4_K_M.gguf",
            "loaded": True,
        },
    ],
    ids=["inactive", "active"],
)
def test_omitted_provider_id_uses_inference_engine_not_manager_load(
    client: Any,
    fts_root: Path,
    monkeypatch: pytest.MonkeyPatch,
    mgr_active: dict[str, Any] | None,
) -> None:
    """Inference loaded + other provider inactive/active → engine, no load."""
    pid = _project(client)
    eng = _loaded_engine()
    mgr = _manager_with_other_provider(active=mgr_active)

    monkeypatch.setattr(
        "finetune_studio.models.manager.get_manager",
        lambda: mgr,
    )
    monkeypatch.setattr(
        "finetune_studio.webui.app.inference_engine",
        eng,
    )

    r = client.post(
        f"/api/projects/{pid}/data-prep/chat",
        json={"messages": [{"role": "user", "content": "hi"}]},
    )
    assert r.status_code == 200, r.text
    body = r.json()
    assert body.get("ok") is True
    assert body.get("reply") == "ok from inference engine"
    eng.model.create_chat_completion.assert_called()
    mgr.load.assert_not_called()
    # Prefer inference even when manager.active() is set (E2E-21).
    mgr.chat.assert_not_called()


def test_omitted_provider_id_409_when_nothing_loaded(
    client: Any,
    fts_root: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    pid = _project(client)
    mgr = MagicMock()
    mgr.active.return_value = None
    mgr.load = MagicMock(
        side_effect=AssertionError("manager.load must not be called"),
    )
    eng = MagicMock()
    eng.model = None
    eng.model_path = None

    monkeypatch.setattr(
        "finetune_studio.models.manager.get_manager",
        lambda: mgr,
    )
    monkeypatch.setattr(
        "finetune_studio.webui.app.inference_engine",
        eng,
    )

    r = client.post(
        f"/api/projects/{pid}/data-prep/chat",
        json={"messages": [{"role": "user", "content": "hi"}]},
    )
    assert r.status_code == 409, r.text
    assert r.json().get("error") == NO_MODEL_MSG
    mgr.load.assert_not_called()


def test_resolve_loaded_backend_prefer_inference(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from finetune_studio.data.prep import generator as gen_mod

    eng = _loaded_engine()
    mgr = _manager_with_other_provider(
        active={"id": "local-default", "loaded": True},
    )
    monkeypatch.setattr(
        "finetune_studio.models.manager.get_manager",
        lambda: mgr,
    )
    monkeypatch.setattr(
        "finetune_studio.webui.app.inference_engine",
        eng,
    )

    both = gen_mod.resolve_loaded_backend(prefer_inference=True)
    assert both is not None
    assert both["kind"] == "global"
    assert both["engine"] is eng

    mgr_first = gen_mod.resolve_loaded_backend(prefer_inference=False)
    assert mgr_first is not None
    assert mgr_first["kind"] == "provider"
    assert mgr_first["manager"] is mgr
