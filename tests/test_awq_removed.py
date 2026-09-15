"""Regression: AWQ removed from export UI + rejected by export APIs."""

from __future__ import annotations

from pathlib import Path

import pytest

from finetune_studio.training.run_export import (
    SUPPORTED_EXPORT_FORMATS,
    export_trained_run,
)


def _merged_run(tmp_path: Path) -> dict:
    out = tmp_path / "run"
    merged = out / "merged"
    merged.mkdir(parents=True)
    (merged / "config.json").write_text("{}", encoding="utf-8")
    (merged / "model.safetensors").write_bytes(b"x")
    return {
        "id": "r1",
        "output_path": str(out),
        "base_model": "Qwen/Qwen3-0.6B",
        "status": "done",
    }


def test_supported_formats_exclude_awq() -> None:
    assert "awq" not in SUPPORTED_EXPORT_FORMATS
    assert SUPPORTED_EXPORT_FORMATS == frozenset(
        {"gguf", "gptq", "abliterated", "merged"}
    )


def test_export_trained_run_rejects_awq_with_clear_message(tmp_path: Path) -> None:
    result = export_trained_run(_merged_run(tmp_path), fmt="awq")
    assert result.get("ok") is False
    assert result.get("status") == "failed"
    assert "AWQ" in result["error"]
    assert "gptq" in result["error"].lower() or "gguf" in result["error"].lower()


def test_export_page_has_no_awq_choice(client) -> None:
    pid = client.post("/api/projects", json={"name": "No AWQ UI"}).json()["id"]
    body = client.get(f"/projects/{pid}/export").text
    assert 'value="awq"' not in body
    assert "name=\"export-format\" value=\"awq\"" not in body
    # Must not advertise AWQ as an available choice.
    assert "AWQ is not available" not in body
    assert "autoawq" not in body.lower()


def test_export_api_rejects_awq(client, tmp_path: Path) -> None:
    from finetune_studio import db

    pid = client.post("/api/projects", json={"name": "AWQ API"}).json()["id"]
    run = _merged_run(tmp_path)
    created = db.create_run(
        project_id=pid,
        name="m",
        base_model=run["base_model"],
        data_path="/d",
    )
    db.update_run(created["id"], status="done", output_path=run["output_path"])
    r = client.post(
        f"/api/projects/{pid}/runs/{created['id']}/export",
        json={"format": "awq", "quants": ["q8_0"]},
    )
    assert r.status_code == 400, r.text
    body = r.json()
    assert body.get("ok") is False
    assert "AWQ" in body["error"]


@pytest.mark.parametrize("fmt", sorted(SUPPORTED_EXPORT_FORMATS))
def test_supported_format_names_are_non_awq(fmt: str) -> None:
    assert fmt != "awq"
