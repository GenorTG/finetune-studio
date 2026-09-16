"""Focused tests for the configured 27B GGUF helper defaults."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from finetune_studio.models.helper import (
    DEFAULT_HELPER_GGUF_BASENAME,
    DEFAULT_HELPER_LABEL,
    DEFAULT_HELPER_PROVIDER_ID,
    annotate_provider,
    default_helper_gguf_path,
    helper_display_label,
    is_helper_gguf_path,
    is_helper_provider,
    no_helper_message,
    wrong_model_message,
)
from finetune_studio.models.providers import PROVIDER_PRESETS
from finetune_studio.models.registry import _safe_model_name


def test_default_helper_constants_are_explicit() -> None:
    assert DEFAULT_HELPER_PROVIDER_ID == "local-default"
    assert "27B" in DEFAULT_HELPER_LABEL
    assert DEFAULT_HELPER_GGUF_BASENAME.endswith(".gguf")
    assert DEFAULT_HELPER_GGUF_BASENAME in default_helper_gguf_path()


def test_fts_helper_gguf_env_override(
    monkeypatch: Any, tmp_path: Path
) -> None:
    custom = tmp_path / "custom-helper.gguf"
    monkeypatch.setenv("FTS_HELPER_GGUF", str(custom))
    # Re-import path helper after env set — function reads env each call.
    assert default_helper_gguf_path() == str(custom)


def test_is_helper_gguf_path_by_basename() -> None:
    assert is_helper_gguf_path(f"/any/where/{DEFAULT_HELPER_GGUF_BASENAME}")
    assert not is_helper_gguf_path("/models/merged/Qwen3-4B")


def test_is_helper_provider_by_id_and_path() -> None:
    assert is_helper_provider({"id": DEFAULT_HELPER_PROVIDER_ID, "model_id": ""})
    assert is_helper_provider(
        {"id": "other", "model_id": f"/x/{DEFAULT_HELPER_GGUF_BASENAME}"}
    )
    assert not is_helper_provider(
        {"id": "other", "model_id": "/models/Qwen3-4B"}
    )


def test_annotate_provider_renames_legacy_local_gguf() -> None:
    row = annotate_provider(
        {
            "id": DEFAULT_HELPER_PROVIDER_ID,
            "name": "Local GGUF",
            "model_id": default_helper_gguf_path(),
            "kind": "local_gguf",
        }
    )
    assert row["is_helper"] is True
    assert row["name"] == DEFAULT_HELPER_LABEL
    assert row["label"] == DEFAULT_HELPER_LABEL


def test_helper_display_label_prefixes() -> None:
    assert helper_display_label(name="Local GGUF").startswith("Helper")
    assert "27B" in helper_display_label(model_id=DEFAULT_HELPER_GGUF_BASENAME)


def test_wrong_and_no_helper_messages_name_provider() -> None:
    assert DEFAULT_HELPER_PROVIDER_ID in no_helper_message()
    assert DEFAULT_HELPER_LABEL in no_helper_message()
    msg = wrong_model_message("/models/Qwen3-4B")
    assert "Qwen3-4B" in msg
    assert DEFAULT_HELPER_LABEL in msg


def test_provider_presets_lead_with_helper() -> None:
    assert PROVIDER_PRESETS[0]["id"] == DEFAULT_HELPER_PROVIDER_ID
    assert PROVIDER_PRESETS[0]["name"] == DEFAULT_HELPER_LABEL
    assert DEFAULT_HELPER_GGUF_BASENAME in PROVIDER_PRESETS[0]["model_id"]


def test_registry_generic_dirs_exclude_awq() -> None:
    # _safe_model_name treats only active export dirs as generic.
    name = _safe_model_name("/proj/output/awq", {}, project_name="Demo")
    # Without awq in generic_dirs, dirname "awq" is returned as-is (not Demo (awq)).
    assert name == "awq"


def test_providers_api_exposes_helper(client: Any) -> None:
    r = client.get("/api/providers")
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["helper_provider_id"] == DEFAULT_HELPER_PROVIDER_ID
    assert "27B" in body["helper_label"]
    assert any(p.get("id") == DEFAULT_HELPER_PROVIDER_ID for p in body["providers"])


def test_data_prep_page_shows_helper(client: Any) -> None:
    pid = client.post("/api/projects", json={"name": "Helper Prep UI"}).json()["id"]
    r = client.get(f"/projects/{pid}/data-prep")
    assert r.status_code == 200, r.text
    assert DEFAULT_HELPER_LABEL in r.text or "27B" in r.text
    assert DEFAULT_HELPER_PROVIDER_ID in r.text
    # Prep job copy: helper-only, load that provider first (no silent fallback).
    assert "load that provider first" in r.text
    assert "Uses" in r.text and "only" in r.text.lower()


def test_testing_page_shows_helper(client: Any) -> None:
    pid = client.post("/api/projects", json={"name": "Helper Test UI"}).json()["id"]
    r = client.get(f"/projects/{pid}/testing")
    assert r.status_code == 200, r.text
    assert "helper_label" not in r.text  # rendered, not the key
    assert "27B" in r.text or "Helper" in r.text
    assert "silent" in r.text.lower() or "never a silent" in r.text


def test_auto_suite_generate_names_helper(
    client: Any, tmp_path: Path, monkeypatch: Any
) -> None:
    from finetune_studio import db

    pid = client.post("/api/projects", json={"name": "Suite Helper"}).json()["id"]
    data = tmp_path / "train.jsonl"
    data.write_text(
        '{"conversations":[{"from":"human","value":"Q?"},{"from":"gpt","value":"A"}]}\n',
        encoding="utf-8",
    )
    out = tmp_path / "out"
    out.mkdir()
    run = db.create_run(
        project_id=pid,
        name="r1",
        base_model="x/y",
        data_path=str(data),
    )
    db.update_run(run["id"], status="done", output_path=str(out))

    r = client.post(f"/api/training/runs/{run['id']}/auto-suites/generate")
    assert r.status_code == 200, r.text
    body = r.json()
    assert body.get("ok") is True
    assert body.get("helper_provider_id") == DEFAULT_HELPER_PROVIDER_ID
    assert "27B" in (body.get("helper_label") or "")
    assert body.get("generation_mode") == "deterministic"


def test_architecture_docs_activity_is_sse() -> None:
    text = Path("docs/ARCHITECTURE.md").read_text(encoding="utf-8")
    assert "drawer polls every 2s" not in text
    assert "/api/activity/events" in text
    assert "SSE" in text
