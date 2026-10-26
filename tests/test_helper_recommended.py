"""Settings -> Helper model: the recommended local helper, its one-click download, and picking an installed GGUF."""
from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest
from fastapi.testclient import TestClient

from finetune_studio.models import helper as helper_mod


def test_recommended_helper_is_googles_qat_build_and_the_default_path() -> None:
    rec = helper_mod.RECOMMENDED_LOCAL_HELPER
    assert rec.repo_id == "google/gemma-4-12B-it-qat-q4_0-gguf" and rec.license == "Apache-2.0"
    assert len(rec.revision) == 40  # immutable Hub commit, not the moving main branch
    gguf, mmproj = helper_mod.recommended_helper_paths()
    assert gguf.name == rec.gguf and mmproj.name == rec.mmproj and gguf.parent.name == "google__gemma-4-12B-it-qat-q4_0-gguf"
    assert helper_mod.DEFAULT_HELPER_GGUF_BASENAME == rec.gguf


def test_the_previous_default_helper_file_is_still_recognised() -> None:
    assert helper_mod.is_helper_gguf_path("/m/gemma-4-12b-it-uncensored-Q4_K_M.gguf")


def test_missing_helper_message_points_at_the_one_click_download() -> None:
    msg = helper_mod.helper_missing_message("/x/y.gguf")
    assert "Settings -> AI assistant" in msg and helper_mod.RECOMMENDED_LOCAL_HELPER.label in msg


def test_settings_state_reports_the_local_file_and_the_recommendation(client: TestClient, monkeypatch, tmp_path: Path) -> None:
    from finetune_studio.webui import app as webapp
    g = tmp_path / "a-model.gguf"
    g.write_bytes(b"GGUF")
    (tmp_path / "mmproj-a.gguf").write_bytes(b"GGUF")
    monkeypatch.setattr(webapp, "discovered_models", [
        SimpleNamespace(path=str(g), format="gguf", size_gb=0.1),
        SimpleNamespace(path=str(tmp_path / "mmproj-a.gguf"), format="gguf", size_gb=0.1),
    ])
    configured = client.post("/api/settings/connections", json={
        "name": "Placeholder", "base_url": "https://example.invalid/v1", "model": "placeholder", "api_key": "not-a-real-key",
    })
    assert configured.status_code == 200
    client.put("/api/settings/assistant", json={"provider_id": configured.json()["id"]})
    d = client.get("/api/settings/assistant").json()
    local = d["local"]
    assert local["recommended"]["repo_id"] == helper_mod.RECOMMENDED_LOCAL_HELPER.repo_id
    assert [c["name"] for c in local["candidates"]] == ["a-model.gguf"]   # projectors are not models

    r = client.post("/api/settings/helper/local", json={"path": str(g)})
    row = next(c for c in r.json()["connections"] if c["id"] == helper_mod.DEFAULT_HELPER_PROVIDER_ID)
    assert r.status_code == 200 and row["path"] == str(g) and row["present"] is True
    assert row["label"] == "Helper · a-model"
    # Changing the local model does not change who the main assistant is.
    assert r.json()["assistant_provider_id"] == configured.json()["id"]
    bad = client.post("/api/settings/helper/local", json={"path": str(tmp_path / "mmproj-a.gguf")})
    assert bad.status_code == 400 and "projector" in bad.json()["detail"]
    assert client.post("/api/settings/helper/local", json={"path": str(tmp_path / "nope.gguf")}).status_code == 400


@pytest.mark.parametrize("installed", [False, True])
def test_recommended_download_starts_only_missing_files(client: TestClient, monkeypatch, tmp_path: Path,
                                                        installed: bool) -> None:
    from finetune_studio.webui import download_jobs
    gguf, mmproj = tmp_path / "rec" / "m.gguf", tmp_path / "rec" / "mmproj-m.gguf"
    monkeypatch.setattr("finetune_studio.webui.routes.helper_settings.recommended_helper_paths", lambda: (gguf, mmproj))
    if installed:
        gguf.parent.mkdir(parents=True)
        gguf.write_bytes(b"GGUF")
        mmproj.write_bytes(b"GGUF")
    calls = []
    monkeypatch.setattr(download_jobs, "start", lambda **kw: calls.append(kw) or {"id": f"job{len(calls)}"})
    d = client.post("/api/settings/helper/recommended/download").json()
    if installed:
        assert d["already_installed"] and not calls
    else:
        assert [c["filename"] for c in calls] == ["m.gguf", "mmproj-m.gguf"]
        assert {c["dest"] for c in calls} == {str(gguf.parent)} and d["path"] == str(gguf)
        assert {c["revision"] for c in calls} == {helper_mod.RECOMMENDED_LOCAL_HELPER.revision}
