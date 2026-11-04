"""Model Library → Downloaded files: a search result that is not downloaded lists no files instead of failing.

Found by the control sweep: every not-downloaded card's "Downloaded files" button got a 404 and the modal said
"Failed: not found" instead of "No files downloaded yet".
"""
from __future__ import annotations

from pathlib import Path


def test_not_downloaded_model_lists_no_files(client, monkeypatch, tmp_path: Path) -> None:
    from finetune_studio.webui.routes import hf_models

    monkeypatch.setattr(hf_models, "_LOCAL", tmp_path)
    r = client.get("/api/hf/local/Org/Nope/files")
    assert r.status_code == 200
    assert r.json()["files"] == [] and r.json()["downloaded"] is False


def test_downloaded_model_lists_its_files(client, monkeypatch, tmp_path: Path) -> None:
    from finetune_studio.webui.routes import hf_models

    monkeypatch.setattr(hf_models, "_LOCAL", tmp_path)
    (tmp_path / "Org__Tiny").mkdir()
    (tmp_path / "Org__Tiny" / "config.json").write_text("{}")
    d = client.get("/api/hf/local/Org/Tiny/files").json()
    assert d["downloaded"] is True and [f["path"] for f in d["files"]] == ["config.json"]


def test_bad_repo_id_is_still_refused(client) -> None:
    assert client.get("/api/hf/local/Org/bad%20name/files").status_code == 400


def test_ui_shows_the_empty_state_for_no_files() -> None:
    page = (Path(__file__).resolve().parents[1] / "src/finetune_studio/webui/templates/hf_models.html").read_text()
    body = page[page.index("async function showFiles"):][:1200]
    assert "/files`" in body and "No files downloaded yet." in body
