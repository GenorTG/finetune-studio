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
from tests.testing_run_support import FakeEngine, install_engine, wait_run_finished


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


def test_cases_from_training_jsonl_selects_user_and_assistant_roles(
    tmp_path: Path,
) -> None:
    p = tmp_path / "system-prefixed.jsonl"
    p.write_text(
        json.dumps(
            {
                "conversations": [
                    {"from": "system", "value": "Answer briefly."},
                    {"from": "human", "value": "What is 2 + 2?"},
                    {"from": "gpt", "value": "4"},
                ]
            }
        )
        + "\n",
        encoding="utf-8",
    )
    cases, skipped = cases_from_training_jsonl(str(p))
    assert skipped == 0
    assert len(cases) == 1
    assert cases[0].question == "What is 2 + 2?"
    assert cases[0].correct_answer == "4"


def test_cases_from_training_jsonl_skips_malformed_message_entries(
    tmp_path: Path,
) -> None:
    p = tmp_path / "malformed.jsonl"
    p.write_text(
        json.dumps({"messages": ["bad-entry", {"role": "user", "content": "Q?"}]})
        + "\n",
        encoding="utf-8",
    )
    cases, skipped = cases_from_training_jsonl(str(p))
    assert cases == []
    assert skipped == 1


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


def test_evaluate_training_api_starts_a_run_that_saves_per_case_transcripts_without_verdicts(
    client, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    proj = db.create_project(name="tev-api", base_model="x/y")
    p = tmp_path / "approved.jsonl"
    _write_sharegpt(p, [
        ("What color is the sky on a clear day?", "blue"),
        ("2+2?", "4"),
    ])
    ds = datasets_db.create_dataset(
        proj["id"], "sky", str(p), source="data-prep-export", qa_count=2
    )

    def answer(q: str) -> str:
        return "The sky is blue" if "sky" in q.lower() else "4"

    install_engine(monkeypatch, FakeEngine(answer, model_path="/fake/model"))

    r = client.post(
        "/api/testing/evaluate-training",
        json={"project_id": proj["id"], "dataset_id": ds["id"], "max_cases": 10, "model_path": "/fake/model"},
    )
    assert r.status_code == 202, r.text
    started = r.json()
    assert started["ok"] is True and started["benchmark"]["kind"] == "training_leakage"
    assert started["benchmark"]["progress_total"] == 2
    done = wait_run_finished(started["benchmark_id"])

    assert done["status"] == "done"
    assert done["scores"]["eval_kind"] == "training_leakage"
    assert "memorization" in done["scores"]["leakage_warning"].lower() or "leakage" in done["scores"]["leakage_warning"].lower()
    assert done["scores"]["dataset_id"] == ds["id"] and done["config"]["eval"]["eval_kind"] == "training_leakage"
    assert done["scores"]["total"] == 2 and done["scores"]["judged"] == 0 and done["scores"]["pass_rate"] is None

    cases = client.get(f"/api/testing/projects/{proj['id']}/runs/{started['benchmark_id']}/cases").json()
    assert len(cases) == 2
    assert cases[0]["model_answer"] and cases[0]["correct_answer"] and cases[0]["verdict"] == ""
    assert "source_id" in cases[0] and "chunk_idx" in cases[0]
    full = client.get(
        f"/api/testing/projects/{proj['id']}/runs/{started['benchmark_id']}/cases/{cases[0]['id']}"
    ).json()
    assert full["transcript"]


def test_evaluate_training_api_heldout_kind_is_recorded(
    client, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    proj = db.create_project(name="tev-heldout", base_model="x/y")
    p = tmp_path / "approved.jsonl"
    _write_sharegpt(p, [(f"Q{i}?", f"A{i}") for i in range(20)])
    ds = datasets_db.create_dataset(proj["id"], "approved", str(p), source="data-prep-export", qa_count=20)
    install_engine(monkeypatch, FakeEngine(lambda q: "x", model_path="/fake/model"))
    r = client.post(
        "/api/testing/evaluate-training",
        json={"project_id": proj["id"], "dataset_id": ds["id"], "eval_kind": "heldout", "model_path": "/fake/model"},
    )
    assert r.status_code == 202, r.text
    assert r.json()["benchmark"]["kind"] == "heldout"
    done = wait_run_finished(r.json()["benchmark_id"])
    assert done["scores"]["eval_kind"] == "heldout" and done["scores"]["total"] == 2
