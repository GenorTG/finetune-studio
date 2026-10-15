"""build_preference_dataset: file + registry contract, and proof the Training route accepts the output unchanged."""
from __future__ import annotations

import json
from pathlib import Path

import pytest

pytest_plugins = ("tests.test_training_start_guard",)

from finetune_studio import db
from finetune_studio.data.prep import preference as pref
from finetune_studio.db.datasets import datasets_dir, list_datasets
from finetune_studio.training.data import format_for_preference, load_jsonl
from tests.preference_fakes import FakeModel, seed_project
from tests.test_training_start_guard import _project, _start


def test_build_writes_the_file_and_registers_it_like_an_upload() -> None:
    proj = seed_project()
    built = pref.build_preference_dataset(proj["id"], max_pairs=6, seed=5, generate=FakeModel())
    path = datasets_dir(proj["id"]) / f"{proj['id']}-preference.jsonl"
    assert built.persisted.path == path and path.is_file()
    rows = load_jsonl(str(path))
    assert len(rows) == built.report.total == 6
    (ds,) = list_datasets(proj["id"])
    assert ds["data_path"] == str(path) and ds["qa_count"] == 6 and ds["source"] == "data-prep-preference"
    assert ds["name"] == "pref-docs · preference · 6 pairs"


def test_rebuilding_updates_the_same_registry_row() -> None:
    proj = seed_project()
    for n in (4, 6):
        pref.build_preference_dataset(proj["id"], max_pairs=n, generate=FakeModel())
    (ds,) = list_datasets(proj["id"])
    assert ds["qa_count"] == 6 and ds["name"].endswith("6 pairs")


def test_no_approved_pairs_is_an_honest_error_and_writes_nothing() -> None:
    proj = seed_project(status="pending")
    with pytest.raises(pref.NoApprovedPairs):
        pref.build_preference_dataset(proj["id"], generate=FakeModel())
    assert not list(datasets_dir(proj["id"]).glob("*.jsonl")) and list_datasets(proj["id"]) == []


def test_without_a_model_nothing_is_fabricated(monkeypatch: pytest.MonkeyPatch) -> None:
    proj = seed_project()
    monkeypatch.setattr("finetune_studio.data.prep.generator.resolve_generator", lambda: None)
    with pytest.raises(pref.NoGenerator):
        pref.build_preference_dataset(proj["id"])
    assert list_datasets(proj["id"]) == []


def test_every_candidate_dropped_is_reported_not_written() -> None:
    proj = seed_project()
    with pytest.raises(pref.NoUsablePairs) as exc:
        pref.build_preference_dataset(proj["id"], generate=FakeModel(hallucination="I don't know."),
                                      kinds=("hallucination",))
    assert exc.value.report.dropped["hallucination"]
    assert list_datasets(proj["id"]) == []


def test_default_generate_is_the_loaded_helper(monkeypatch: pytest.MonkeyPatch) -> None:
    proj = seed_project()
    fake = FakeModel()
    monkeypatch.setattr(
        "finetune_studio.data.prep.generator.resolve_generator",
        lambda: lambda messages, **kw: fake(messages[-1]["content"]),
    )
    built = pref.build_preference_dataset(proj["id"], max_pairs=4)
    assert built.report.total == 4 and fake.calls


# ── compatibility: the existing Training path accepts the file unchanged ──

def test_output_passes_the_validator_and_the_training_start_endpoint(client, fake_engine, fake_home,
                                                                     tmp_path: Path) -> None:
    proj = seed_project()
    built = pref.build_preference_dataset(proj["id"], max_pairs=8, generate=FakeModel())
    path = str(built.persisted.path)
    assert len(format_for_preference(load_jsonl(path))) == 8  # existing validator, no adaptation

    model = tmp_path / "model"
    model.mkdir()
    pid = _project(client)
    resp = _start(client, pid, path, str(model), training_mode="dpo")
    assert resp.status_code == 200, resp.text
    assert fake_engine.started["config"].training_mode == "dpo"
    started = fake_engine.started["data"]
    assert len(started) == 8 and {"prompt", "chosen", "rejected"} <= set(started[0])


def test_health_check_and_upload_path_agree_with_the_builder_output(client, fake_home) -> None:
    proj = seed_project()
    built = pref.build_preference_dataset(proj["id"], max_pairs=10, generate=FakeModel())
    body = built.persisted.path.read_bytes()
    pid = _project(client)
    up = client.post(f"/api/projects/{pid}/datasets/upload",
                     files={"file": ("pref.jsonl", body, "application/jsonl")})
    assert up.status_code == 200, up.text
    health = client.get(f"/api/projects/{pid}/datasets/{up.json()['id']}/health?training_mode=dpo")
    assert health.status_code == 200, health.text
    assert health.json()["trainable"] == health.json()["examples"] == 10
    kept = [json.loads(line) for line in built.persisted.path.read_text().splitlines()]
    assert all(r["meta"]["origin"] == "helper-model" for r in kept)


def test_unknown_project_has_no_pairs() -> None:
    assert db.get_project("nope") is None
    with pytest.raises(pref.NoApprovedPairs):
        pref.build_preference_dataset("nope", generate=FakeModel())
