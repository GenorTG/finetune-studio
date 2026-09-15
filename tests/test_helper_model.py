"""Regression: configured 27B GGUF helper is explicit for data-prep / suites."""

from __future__ import annotations

from pathlib import Path
from typing import Any
from unittest.mock import MagicMock

import pytest

from finetune_studio.models.helper import (
    DEFAULT_HELPER_GGUF_BASENAME,
    DEFAULT_HELPER_LABEL,
    DEFAULT_HELPER_PROVIDER_ID,
    annotate_provider,
    helper_display_label,
    is_helper_gguf_path,
    is_helper_provider,
    no_helper_message,
    wrong_model_message,
)


def test_helper_constants_and_label() -> None:
    assert DEFAULT_HELPER_PROVIDER_ID == "local-default"
    assert "27B" in DEFAULT_HELPER_LABEL
    assert DEFAULT_HELPER_LABEL.startswith("Helper")
    assert helper_display_label(name="Local GGUF").startswith("Helper ·")
    assert is_helper_gguf_path(
        f"/home/x/finetune-studio/models/gguf/{DEFAULT_HELPER_GGUF_BASENAME}"
    )
    assert not is_helper_gguf_path("/models/merged/Qwen3-4B")


def test_annotate_provider_marks_helper() -> None:
    row = annotate_provider({
        "id": DEFAULT_HELPER_PROVIDER_ID,
        "name": "Local GGUF",
        "kind": "local_gguf",
        "model_id": f"/m/{DEFAULT_HELPER_GGUF_BASENAME}",
    })
    assert row["is_helper"] is True
    assert row["label"].startswith("Helper")
    assert row["name"] == DEFAULT_HELPER_LABEL
    assert is_helper_provider(row)


def test_providers_api_exposes_helper(client: Any) -> None:
    r = client.get("/api/providers")
    assert r.status_code == 200, r.text
    body = r.json()
    assert body.get("helper_provider_id") == DEFAULT_HELPER_PROVIDER_ID
    assert "Helper" in (body.get("helper_label") or "")
    assert "providers" in body
    helpers = [p for p in body["providers"] if p.get("is_helper")]
    assert helpers, "seeded local-default helper must be present"
    assert helpers[0]["id"] == DEFAULT_HELPER_PROVIDER_ID


def test_data_prep_page_shows_helper_label(client: Any) -> None:
    pid = client.post("/api/projects", json={"name": "Helper UI"}).json()["id"]
    r = client.get(f"/projects/{pid}/data-prep")
    assert r.status_code == 200
    assert DEFAULT_HELPER_LABEL in r.text or "Helper ·" in r.text
    assert DEFAULT_HELPER_PROVIDER_ID in r.text
    assert "will not" in r.text.lower() or "silently" in r.text.lower()


def test_testing_page_shows_helper_label(client: Any) -> None:
    pid = client.post("/api/projects", json={"name": "Helper Testing"}).json()["id"]
    r = client.get(f"/projects/{pid}/testing")
    assert r.status_code == 200
    assert "Helper" in r.text
    assert "silent" in r.text.lower()


def test_resolve_generator_rejects_non_helper_inference(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from finetune_studio.data.prep import generator as gen_mod

    mgr = MagicMock()
    mgr.active.return_value = None
    eng = MagicMock()
    eng.model = object()
    eng.model_path = "/models/merged/project-4b"

    monkeypatch.setattr(
        "finetune_studio.models.manager.get_manager",
        lambda: mgr,
    )
    monkeypatch.setattr(
        "finetune_studio.webui.app.inference_engine",
        eng,
    )

    assert gen_mod.resolve_generator() is None
    err = gen_mod.helper_resolution_error()
    assert DEFAULT_HELPER_LABEL in err or "helper" in err.lower()
    assert "4b" in err.lower() or "project-4b" in err or "merged" in err


def test_resolve_generator_accepts_helper_on_inference(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from finetune_studio.data.prep import generator as gen_mod

    helper_path = f"/models/gguf/{DEFAULT_HELPER_GGUF_BASENAME}"
    mgr = MagicMock()
    mgr.active.return_value = None
    eng = MagicMock()
    eng.model = object()
    eng.model_path = helper_path
    eng.generate.return_value = "ok"

    monkeypatch.setattr(
        "finetune_studio.models.manager.get_manager",
        lambda: mgr,
    )
    monkeypatch.setattr(
        "finetune_studio.webui.app.inference_engine",
        eng,
    )

    chat = gen_mod.resolve_generator()
    assert chat is not None
    assert chat([{"role": "user", "content": "hi"}]) == "ok"
    eng.generate.assert_called_once()


def test_resolve_generator_accepts_helper_manager_active(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from finetune_studio.data.prep import generator as gen_mod

    mgr = MagicMock()
    mgr.active.return_value = {
        "id": DEFAULT_HELPER_PROVIDER_ID,
        "kind": "local_gguf",
        "model_id": f"/x/{DEFAULT_HELPER_GGUF_BASENAME}",
        "name": DEFAULT_HELPER_LABEL,
        "loaded": True,
    }
    mgr.chat.return_value = "from-helper"
    eng = MagicMock()
    eng.model = object()
    eng.model_path = "/other/model.gguf"  # non-helper — must not be used

    monkeypatch.setattr(
        "finetune_studio.models.manager.get_manager",
        lambda: mgr,
    )
    monkeypatch.setattr(
        "finetune_studio.webui.app.inference_engine",
        eng,
    )

    chat = gen_mod.resolve_generator()
    assert chat is not None
    assert chat([{"role": "user", "content": "hi"}]) == "from-helper"
    mgr.chat.assert_called_once()
    eng.generate.assert_not_called()


def test_omitted_provider_id_409_when_non_helper_loaded(
    client: Any,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = tmp_path / "fts"
    root.mkdir()
    projects = root / "projects"
    projects.mkdir()
    monkeypatch.setattr("finetune_studio.data.fs.paths._ROOT", root)
    monkeypatch.setattr("finetune_studio.data.fs.paths._PROJECTS", projects)

    pid = client.post("/api/projects", json={"name": "No Silent Fallback"}).json()["id"]
    mgr = MagicMock()
    mgr.active.return_value = {
        "id": "other",
        "kind": "local_gguf",
        "model_id": "/models/other.gguf",
        "loaded": True,
    }
    eng = MagicMock()
    eng.model = object()
    eng.model_path = "/models/other.gguf"

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
    assert "helper" in err.lower()
    assert wrong_model_message("/models/other.gguf").split(",")[0] in err or "other" in err


def test_architecture_doc_describes_activity_sse() -> None:
    arch = Path("docs/ARCHITECTURE.md").read_text(encoding="utf-8")
    assert "polls every 2s" not in arch
    assert "/api/activity/events" in arch or "SSE" in arch
    assert "activity" in arch.lower()


def test_registry_generic_dirs_exclude_awq() -> None:
    from finetune_studio.models import registry as reg

    src = Path(reg.__file__).read_text(encoding="utf-8")
    # Active naming lists must not treat AWQ as a current export format.
    assert '"awq"' not in src and "'awq'" not in src


def test_no_helper_message_names_label() -> None:
    msg = no_helper_message()
    assert DEFAULT_HELPER_LABEL in msg
    assert DEFAULT_HELPER_PROVIDER_ID in msg
