"""Tests for POST /api/projects/{pid}/data-prep/start (E2E-9).

Covers generator resolution: configured helper on manager / Inference,
rejection of non-helper loads, and none-loaded 409.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any
from unittest.mock import MagicMock

import pytest

from finetune_studio.data.fs import file_library as fl
from finetune_studio.data.prep.generator import HELPER_NO_MODEL_MSG
from finetune_studio.models.helper import (
    DEFAULT_HELPER_GGUF_BASENAME,
    DEFAULT_HELPER_LABEL,
    DEFAULT_HELPER_PROVIDER_ID,
)


@pytest.fixture
def fts_root(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Point the file-library disk root at a temp dir for isolated tests."""
    root = tmp_path / "fts"
    root.mkdir()
    projects = root / "projects"
    projects.mkdir()
    monkeypatch.setattr("finetune_studio.data.fs.paths._ROOT", root)
    monkeypatch.setattr("finetune_studio.data.fs.paths._PROJECTS", projects)
    fl._PARSED_CACHE.clear()
    return root


def _project(client: Any) -> str:
    r = client.post("/api/projects", json={"name": "Data Prep Start Test"})
    assert r.status_code in (200, 201), r.text
    return r.json()["id"]


def _upload_and_promote(client: Any, pid: str) -> str:
    import secrets

    uniq = secrets.token_hex(8)
    body = (
        f"# Velmaris Ember College ({uniq})\n\n"
        "The capital of Velmaris is Eldrathane.\n"
        "The Ember College teaches pyromancy to apprentices.\n"
    ).encode()
    up = client.post(
        f"/api/projects/{pid}/files/upload",
        files=[("files", (f"velmaris-{uniq}.md", body, "text/markdown"))],
    )
    assert up.status_code == 200, up.text
    report = up.json()["report"][0]
    assert report.get("status") == "uploaded" or "file_id" in report, report
    file_id = report["file_id"]
    promo = client.post(
        f"/api/projects/{pid}/data-prep/sources",
        json={"file_id": file_id},
    )
    assert promo.status_code == 200, promo.text
    return promo.json()["source"]["id"]


def _start_body(source_id: str, **overrides: Any) -> dict[str, Any]:
    body: dict[str, Any] = {
        "source_id": source_id,
        "qa_per_chunk": 3,
        "difficulty": "medium",
        "style": "socratic",
    }
    body.update(overrides)
    return body


def _empty_manager() -> MagicMock:
    mgr = MagicMock()
    mgr.active.return_value = None
    return mgr


def _helper_manager() -> MagicMock:
    mgr = MagicMock()
    mgr.active.return_value = {
        "id": DEFAULT_HELPER_PROVIDER_ID,
        "kind": "local_gguf",
        "model_id": f"/models/gguf/{DEFAULT_HELPER_GGUF_BASENAME}",
        "name": DEFAULT_HELPER_LABEL,
        "loaded": True,
    }
    mgr.chat.return_value = (
        '[{"q":"What is the capital of Velmaris?","a":"Eldrathane"}]'
    )
    return mgr


def _loaded_helper_engine() -> MagicMock:
    eng = MagicMock()
    eng.model = object()
    eng.model_path = f"/models/gguf/{DEFAULT_HELPER_GGUF_BASENAME}"
    eng.generate.return_value = (
        '[{"q":"What is the capital of Velmaris?","a":"Eldrathane"}]'
    )
    return eng


def _loaded_other_engine() -> MagicMock:
    eng = MagicMock()
    eng.model = object()
    eng.model_path = "/fake/other-model.gguf"
    eng.generate.return_value = "should not be used"
    return eng


def _unloaded_engine() -> MagicMock:
    eng = MagicMock()
    eng.model = None
    eng.model_path = None
    return eng


def test_start_unknown_source_404(client: Any, fts_root: Path) -> None:
    pid = _project(client)
    r = client.post(
        f"/api/projects/{pid}/data-prep/start",
        json=_start_body("does-not-exist"),
    )
    assert r.status_code == 404, r.text
    assert "error" in r.json()


def test_start_no_model_409(
    client: Any, fts_root: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Neither manager nor inference_engine loaded → 409 naming the helper."""
    pid = _project(client)
    source_id = _upload_and_promote(client, pid)

    monkeypatch.setattr(
        "finetune_studio.models.manager.get_manager",
        lambda: _empty_manager(),
    )
    monkeypatch.setattr(
        "finetune_studio.webui.app.inference_engine",
        _unloaded_engine(),
    )
    r = client.post(
        f"/api/projects/{pid}/data-prep/start",
        json=_start_body(source_id),
    )
    assert r.status_code == 409, r.text
    err = r.json()["error"]
    assert err == HELPER_NO_MODEL_MSG
    assert DEFAULT_HELPER_LABEL in err
    assert DEFAULT_HELPER_PROVIDER_ID in err
    assert "provider section" not in err


def test_start_rejects_non_helper_model(
    client: Any, fts_root: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A loaded non-helper model must 409 — no silent fallback."""
    pid = _project(client)
    source_id = _upload_and_promote(client, pid)

    monkeypatch.setattr(
        "finetune_studio.models.manager.get_manager",
        lambda: _empty_manager(),
    )
    monkeypatch.setattr(
        "finetune_studio.webui.app.inference_engine",
        _loaded_other_engine(),
    )
    r = client.post(
        f"/api/projects/{pid}/data-prep/start",
        json=_start_body(source_id),
    )
    assert r.status_code == 409, r.text
    err = r.json()["error"]
    assert "helper" in err.lower()
    assert "other-model" in err or "not" in err.lower()


def test_start_ok_when_helper_manager_active(
    client: Any, fts_root: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Helper active on manager is enough even if inference_engine is empty."""
    pid = _project(client)
    source_id = _upload_and_promote(client, pid)

    monkeypatch.setattr(
        "finetune_studio.models.manager.get_manager",
        lambda: _helper_manager(),
    )
    monkeypatch.setattr(
        "finetune_studio.webui.app.inference_engine",
        _unloaded_engine(),
    )
    monkeypatch.setattr(
        "finetune_studio.webui.routes.data_prep._run_prep_background",
        lambda *a: None,
    )

    r = client.post(
        f"/api/projects/{pid}/data-prep/start",
        json=_start_body(
            source_id, qa_per_chunk=5, difficulty="hard", style="factual"
        ),
    )
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["ok"] is True
    assert body["source_id"] == source_id
    run_id = body["run_id"]
    assert run_id

    from finetune_studio.webui.routes import data_prep as dp_mod

    runner = dp_mod._RUNS[(pid, run_id)]["runner"]
    assert runner.qa_per_chunk == 5
    assert runner.difficulty == "hard"
    assert runner.style == "factual"


def test_start_ok_when_helper_on_inference(
    client: Any, fts_root: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Manager inactive + helper GGUF on inference_engine → 200."""
    pid = _project(client)
    source_id = _upload_and_promote(client, pid)

    monkeypatch.setattr(
        "finetune_studio.models.manager.get_manager",
        lambda: _empty_manager(),
    )
    monkeypatch.setattr(
        "finetune_studio.webui.app.inference_engine",
        _loaded_helper_engine(),
    )
    monkeypatch.setattr(
        "finetune_studio.webui.routes.data_prep._run_prep_background",
        lambda *a: None,
    )

    r = client.post(
        f"/api/projects/{pid}/data-prep/start",
        json=_start_body(source_id),
    )
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["ok"] is True
    assert body["source_id"] == source_id
    assert body["run_id"]


def test_start_invalid_difficulty_422(client: Any, fts_root: Path) -> None:
    pid = _project(client)
    r = client.post(
        f"/api/projects/{pid}/data-prep/start",
        json=_start_body("any", difficulty="nightmare"),
    )
    assert r.status_code == 422, r.text


def test_resolve_generator_helper_manager(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from finetune_studio.data.prep import generator as gen_mod

    mgr = _helper_manager()
    eng = _loaded_other_engine()
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
    out = chat(
        [{"role": "user", "content": "hi"}],
        max_tokens=10,
        temperature=0.1,
        top_p=0.5,
    )
    assert "Eldrathane" in out
    mgr.chat.assert_called_once()
    eng.generate.assert_not_called()


def test_resolve_generator_helper_on_engine(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from finetune_studio.data.prep import generator as gen_mod

    mgr = _empty_manager()
    eng = _loaded_helper_engine()
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
    out = chat(
        [{"role": "user", "content": "hi"}],
        max_tokens=10,
        temperature=0.1,
        top_p=0.5,
    )
    assert "Eldrathane" in out
    eng.generate.assert_called_once()
    mgr.chat.assert_not_called()


def test_resolve_generator_none_when_unloaded(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from finetune_studio.data.prep import generator as gen_mod

    monkeypatch.setattr(
        "finetune_studio.models.manager.get_manager",
        lambda: _empty_manager(),
    )
    monkeypatch.setattr(
        "finetune_studio.webui.app.inference_engine",
        _unloaded_engine(),
    )
    assert gen_mod.resolve_generator() is None
