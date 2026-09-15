"""Data-prep chat requires the configured 27B GGUF helper.

When ``provider_id`` is omitted, the route must use the helper already loaded
in ModelManager or on ``inference_engine`` — never a project's merged 4B (or
any other non-helper) and never call ``manager.load``.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any
from unittest.mock import MagicMock

import pytest

from finetune_studio.models.helper import (
    DEFAULT_HELPER_GGUF_BASENAME,
    DEFAULT_HELPER_PROVIDER_ID,
    no_helper_message,
    wrong_model_message,
)

_HELPER_PATH = f"/models/gguf/{DEFAULT_HELPER_GGUF_BASENAME}"


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


def _loaded_engine(
    path: str = _HELPER_PATH,
    reply: str = "ok from inference engine",
) -> MagicMock:
    eng = MagicMock()
    eng.model = MagicMock()
    eng.model_path = path
    eng.model.create_chat_completion.return_value = {
        "choices": [{"message": {"content": reply}}],
    }
    return eng


def _manager_helper(*, active: dict[str, Any] | None) -> MagicMock:
    """Manager that knows the helper provider; load must not run when omitted."""
    mgr = MagicMock()
    mgr.active.return_value = active
    mgr.get_provider.return_value = {
        "id": DEFAULT_HELPER_PROVIDER_ID,
        "kind": "local_gguf",
        "model_id": _HELPER_PATH,
        "name": "Helper · Qwen3.8-27B GGUF",
    }
    mgr.load = MagicMock(
        side_effect=AssertionError("manager.load must not be called"),
    )
    mgr.chat.return_value = "ok from manager"
    return mgr


def test_omitted_provider_id_uses_helper_on_inference(
    client: Any,
    fts_root: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Helper GGUF on Inference + manager inactive → engine, no load."""
    pid = _project(client)
    eng = _loaded_engine()
    mgr = _manager_helper(active=None)

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
    assert "ok from inference engine" in (r.json().get("reply") or "")
    mgr.load.assert_not_called()


def test_omitted_provider_id_uses_helper_manager_when_active(
    client: Any,
    fts_root: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Helper already active in manager → manager chat, no load."""
    pid = _project(client)
    eng = _loaded_engine(path="/models/Qwen3-4B")  # non-helper on Inference
    mgr = _manager_helper(
        active={
            "id": DEFAULT_HELPER_PROVIDER_ID,
            "kind": "local_gguf",
            "model_id": _HELPER_PATH,
            "loaded": True,
        },
    )

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
    assert r.json().get("reply") == "ok from manager"
    mgr.load.assert_not_called()
    mgr.chat.assert_called()


def test_omitted_provider_id_rejects_non_helper_inference(
    client: Any,
    fts_root: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Merged 4B on Inference is not a silent fallback — clear 409."""
    pid = _project(client)
    eng = _loaded_engine(path="/models/Qwen3-4B")
    mgr = _manager_helper(active=None)

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
    err = r.json().get("error") or ""
    assert err == wrong_model_message("/models/Qwen3-4B")
    mgr.load.assert_not_called()


def test_omitted_provider_id_409_when_nothing_loaded(
    client: Any,
    fts_root: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    pid = _project(client)
    eng = MagicMock()
    eng.model = None
    eng.model_path = None
    mgr = _manager_helper(active=None)

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
    assert r.json().get("error") == no_helper_message()
    mgr.load.assert_not_called()


def test_resolve_loaded_backend_is_helper_only(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """resolve_loaded_backend never returns a non-helper Inference load."""
    from finetune_studio.data.prep import generator as gen_mod

    eng = _loaded_engine(path="/models/Qwen3-4B")
    mgr = _manager_helper(
        active={
            "id": DEFAULT_HELPER_PROVIDER_ID,
            "kind": "local_gguf",
            "model_id": _HELPER_PATH,
            "loaded": True,
        },
    )
    monkeypatch.setattr(
        "finetune_studio.models.manager.get_manager",
        lambda: mgr,
    )
    monkeypatch.setattr(
        "finetune_studio.webui.app.inference_engine",
        eng,
    )

    # prefer_inference ignored — helper manager wins; 4B engine is not used.
    both = gen_mod.resolve_loaded_backend(prefer_inference=True)
    assert both is not None
    assert both["kind"] == "provider"
    assert both["manager"] is mgr

    only_other = gen_mod.resolve_loaded_backend(prefer_inference=False)
    assert only_other is not None
    assert only_other["kind"] == "provider"
