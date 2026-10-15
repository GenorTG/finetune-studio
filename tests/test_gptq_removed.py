"""Regression: GPTQ removed from export UI + rejected by export APIs."""

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


def test_supported_formats_exclude_gptq() -> None:
    assert "gptq" not in SUPPORTED_EXPORT_FORMATS
    assert SUPPORTED_EXPORT_FORMATS == frozenset(
        {"gguf", "abliterated", "merged"}
    )


def test_export_trained_run_rejects_gptq_with_clear_message(tmp_path: Path) -> None:
    result = export_trained_run(_merged_run(tmp_path), fmt="gptq")
    assert result.get("ok") is False
    assert result.get("status") == "failed"
    assert "GPTQ" in result["error"]
    assert "gguf" in result["error"].lower()


def test_export_page_has_no_gptq_choice(client) -> None:
    pid = client.post("/api/projects", json={"name": "No GPTQ UI"}).json()["id"]
    body = client.get(f"/projects/{pid}/export").text
    assert 'value="gptq"' not in body
    assert "name=\"export-format\" value=\"gptq\"" not in body
    # Must not advertise GPTQ as an available choice.
    assert "GPTQ is not available" not in body
    assert "gptqmodel" not in body.lower()
    assert "auto_gptq" not in body.lower()


def test_export_api_rejects_gptq(client, tmp_path: Path) -> None:
    from finetune_studio import db

    pid = client.post("/api/projects", json={"name": "GPTQ API"}).json()["id"]
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
        json={"format": "gptq", "quants": ["q8_0"]},
    )
    assert r.status_code == 400, r.text
    body = r.json()
    assert body.get("ok") is False
    assert "GPTQ" in body["error"]


@pytest.mark.parametrize("fmt", sorted(SUPPORTED_EXPORT_FORMATS))
def test_supported_format_names_are_non_gptq(fmt: str) -> None:
    assert fmt != "gptq"
