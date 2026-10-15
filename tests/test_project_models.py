"""Tests for project models expand-row + export contents API."""
from __future__ import annotations

import os
from pathlib import Path
from unittest.mock import MagicMock, patch


def _project(client) -> str:
    r = client.post("/api/projects", json={"name": "Models Expand Test"})
    assert r.status_code in (200, 201), r.text
    return r.json()["id"]


def _run_with_export(
    pid: str, tmp_path: Path, *, name: str = "exp-run"
) -> tuple[dict, Path]:
    """Create a done run whose output_path/merged/ has a couple of files."""
    from finetune_studio import db

    out = tmp_path / "run-out"
    merged = out / "merged"
    merged.mkdir(parents=True)
    (merged / "model.safetensors").write_bytes(b"x" * 100)
    (merged / "config.json").write_text('{"a":1}', encoding="utf-8")
    (merged / "tokenizer.json").write_text("{}", encoding="utf-8")

    run = db.create_run(
        project_id=pid,
        name=name,
        base_model="/m",
        data_path="/d",
        settings_obj={
            "learning_rate": 8e-5,
            "lora_rank": 64,
            "batch_size": 2,
            "num_epochs": 4,
            "max_seq_length": 2048,
            "merge_on_save": True,
        },
    )
    db.update_run(
        run["id"],
        status="done",
        output_path=str(out),
        started_at=1.0,
        finished_at=61.0,
    )
    got = db.get_run(run["id"])
    assert got is not None
    return got, merged


def test_models_page_renders_trained_exports_table(
    client, tmp_path: Path
) -> None:
    pid = _project(client)
    _run_with_export(pid, tmp_path)
    r = client.get(f"/projects/{pid}/models")
    assert r.status_code == 200
    body = r.text
    assert 'id="trained-exports-table"' in body
    cols = (
        "Name", "Format", "Size", "Source run",
        "Created", "Copy path", "Actions",
    )
    for col in cols:
        assert f">{col}</th>" in body
    assert "Copy path" in body
    assert "▶ Open in inference" in body
    assert "flToggleExportRow" in body
    assert "flOpenInInference" in body
    assert "export-row" in body
    assert "Learning rate" in body
    assert "LoRA rank" in body
    assert "Output dir contents" in body


def test_expand_toggle_js_adds_expanded_class(
    client, tmp_path: Path
) -> None:
    """HTML includes flToggleExportRow that toggles .expanded."""
    pid = _project(client)
    _run_with_export(pid, tmp_path)
    body = client.get(f"/projects/{pid}/models").text
    assert "flToggleExportRow" in body
    assert 'classList.add("expanded")' in body
    assert 'querySelectorAll("tr.export-row.expanded")' in body
    assert "classList.remove" in body


def test_export_contents_endpoint_lists_files(
    client, tmp_path: Path
) -> None:
    pid = _project(client)
    _run, merged = _run_with_export(pid, tmp_path)
    # Path without leading slash — matches how the JS builds the URL.
    rel = str(merged).lstrip("/")
    r = client.get(f"/api/projects/{pid}/models/{rel}/contents")
    assert r.status_code == 200, r.text
    data = r.json()
    assert data["path"] == os.path.normpath(str(merged))
    names = {f["name"] for f in data["files"]}
    assert "model.safetensors" in names
    assert "config.json" in names
    assert "tokenizer.json" in names
    by_name = {f["name"]: f for f in data["files"]}
    assert by_name["model.safetensors"]["size_bytes"] == 100
    assert by_name["model.safetensors"]["is_dir"] is False


def test_export_contents_rejects_foreign_path(
    client, tmp_path: Path
) -> None:
    pid = _project(client)
    alien = tmp_path / "alien"
    alien.mkdir()
    (alien / "x.bin").write_bytes(b"hi")
    rel = str(alien).lstrip("/")
    r = client.get(f"/api/projects/{pid}/models/{rel}/contents")
    assert r.status_code == 403


def test_open_in_inference_handler_posts_load(
    client, tmp_path: Path
) -> None:
    """Page JS posts {path} to /api/models/load; API accepts that body."""
    pid = _project(client)
    _run, merged = _run_with_export(pid, tmp_path)
    body = client.get(f"/projects/{pid}/models").text
    assert "/api/models/load" in body
    assert "path: modelPath" in body
    assert 'location.href = "/inference"' in body

    eng = MagicMock()
    eng.load = MagicMock()
    eng.vision = False
    with patch("finetune_studio.webui.app.inference_engine", eng):
        r = client.post("/api/models/load", json={"path": str(merged)})
    assert r.status_code == 200
    data = r.json()
    assert data.get("status") == "loaded" or "error" in data
