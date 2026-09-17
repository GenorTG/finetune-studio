"""Tests for training-data (leakage-aware) evaluation."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from finetune_studio import db
from finetune_studio.config import settings
from finetune_studio.db import datasets as datasets_db
from finetune_studio.testing.training_eval import (
    LEAKAGE_WARNING,
    build_heldout_eval,
    build_training_eval,
    cases_from_training_jsonl,
    suite_label_for_training_eval,
)


@pytest.fixture
def isolated_db(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    db_path = tmp_path / "train_eval.db"
    monkeypatch.setattr(settings, "db_path", str(db_path))
    db.init_db()
    return db_path


def _write_sharegpt(path: Path, pairs: list[tuple[str, str]]) -> None:
    with path.open("w", encoding="utf-8") as f:
        for q, a in pairs:
            f.write(
                json.dumps({
                    "conversations": [
                        {"from": "human", "value": q},
                        {"from": "gpt", "value": a},
                    ]
                })
                + "\n"
            )


def test_cases_from_training_jsonl_sharegpt(tmp_path: Path) -> None:
    p = tmp_path / "ds.jsonl"
    _write_sharegpt(p, [
        ("What is LoRA?", "LoRA adds low-rank adapters."),
        ("Name the capital of France.", "Paris"),
    ])
    cases, skipped = cases_from_training_jsonl(str(p), max_cases=10)
    assert skipped == 0
    assert len(cases) == 2
    assert cases[0].question.startswith("What is LoRA")
    assert "Paris" in cases[1].correct_answer


def test_cases_preserve_source_provenance(tmp_path: Path) -> None:
    p = tmp_path / "provenance.jsonl"
    p.write_text(json.dumps({
        "conversations": [
            {"from": "human", "value": "When?"},
            {"from": "gpt", "value": "2026-10-05"},
        ],
        "source_id": "review.md",
        "chunk_idx": 1,
    }) + "\n", encoding="utf-8")
    cases, skipped = cases_from_training_jsonl(str(p), max_cases=10)
    assert skipped == 0
    assert cases[0].source_id == "review.md"
    assert cases[0].chunk_idx == 1


def test_heldout_eval_is_deterministic_and_excludes_training_slice(
    isolated_db: Path, tmp_path: Path
) -> None:
    proj = db.create_project(name="heldout", base_model="x/y")
    p = tmp_path / "approved.jsonl"
    _write_sharegpt(p, [(f"Q{i}?", f"A{i}") for i in range(20)])
    ds = datasets_db.create_dataset(
        proj["id"], "approved", str(p), source="data-prep-export", qa_count=20
    )
    cases, meta = build_heldout_eval(proj["id"], dataset_id=ds["id"], max_cases=20)
    again, again_meta = build_heldout_eval(proj["id"], dataset_id=ds["id"], max_cases=20)
    assert len(cases) == 2
    assert [c.name for c in cases] == [c.name for c in again]
    assert meta.eval_kind == "heldout"
    assert again_meta.eval_kind == "heldout"
    assert "validation" in meta.leakage_warning.lower()


def test_build_training_eval_meta(isolated_db: Path, tmp_path: Path) -> None:
    proj = db.create_project(name="tev", base_model="x/y")
    p = tmp_path / "approved.jsonl"
    _write_sharegpt(p, [("Q1?", "A1"), ("Q2?", "A2"), ("Q3?", "A3")])
    ds = datasets_db.create_dataset(
        proj["id"], "approved-export", str(p), source="data-prep-export", qa_count=3
    )
    cases, meta = build_training_eval(proj["id"], dataset_id=ds["id"])
    assert len(cases) == 3
    assert meta.eval_kind == "training_leakage"
    assert meta.leakage_warning == LEAKAGE_WARNING
    assert meta.dataset_id == ds["id"]
    assert "training_leakage:" in suite_label_for_training_eval(meta)


def test_evaluate_training_api_returns_per_case_table(
    isolated_db: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from fastapi.testclient import TestClient

    from finetune_studio.webui.app import app

    proj = db.create_project(name="tev-api", base_model="x/y")
    p = tmp_path / "approved.jsonl"
    _write_sharegpt(p, [
        ("What color is the sky on a clear day?", "blue"),
        ("2+2?", "4"),
    ])
    ds = datasets_db.create_dataset(
        proj["id"], "sky", str(p), source="data-prep-export", qa_count=2
    )

    class FakeEngine:
        model = object()
        model_path = "/fake/model"

        def generate(self, messages, max_tokens=512, temperature=0.3, think=False):
            q = messages[0]["content"]
            if "sky" in q.lower():
                return "The sky is blue"
            if "2+2" in q:
                return "4"
            return "unknown"

        def load(self, *a, **k):
            return None

        def unload(self):
            return None

    fake = FakeEngine()
    monkeypatch.setattr(
        "finetune_studio.webui.routes.testing.inference_engine", fake
    )
    monkeypatch.setattr(
        "finetune_studio.webui.app.inference_engine", fake
    )

    client = TestClient(app)
    r = client.post(
        "/api/testing/evaluate-training",
        json={"project_id": proj["id"], "dataset_id": ds["id"], "max_cases": 10},
    )
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["eval"]["eval_kind"] == "training_leakage"
    assert "memorization" in body["eval"]["leakage_warning"].lower() or "leakage" in body["scores"]["leakage_warning"].lower()
    assert body["scores"]["eval_kind"] == "training_leakage"
    assert len(body["results"]) == 2
    assert "verdict" in body["results"][0]
    assert "model_answer" in body["results"][0]
    assert body["scores"]["judged"] >= 1


def test_list_training_datasets_endpoint(isolated_db: Path, tmp_path: Path) -> None:
    from fastapi.testclient import TestClient

    from finetune_studio.webui.app import app

    proj = db.create_project(name="tev-list", base_model="x/y")
    p = tmp_path / "d.jsonl"
    _write_sharegpt(p, [("a?", "b")])
    datasets_db.create_dataset(proj["id"], "d1", str(p), qa_count=1)
    client = TestClient(app)
    r = client.get(f"/api/testing/projects/{proj['id']}/training-datasets")
    assert r.status_code == 200
    data = r.json()
    assert len(data["datasets"]) == 1
    assert "leakage" in data["leakage_warning"].lower()
